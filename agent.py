import os
import subprocess
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import random
import cv2
import datetime
from buffer import ReplayBuffer
from episode_buffer import EpisodeBuffer
from goal_geometry import (world_coords, spin_fraction, spin_thresholds, SPIN_WINDOW,
                           reach_reward, reach_radius_at)
from motion import MotionState, motion_dim
from models.q_model import QModel
from policy import softmax_rel_probs
from task_chain import DEFAULT_CHAIN
from chained_eval import run_chain
from torch.utils.tensorboard.writer import SummaryWriter


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 100000,
                       target_update_interval: int = 1000,
                       goal_layers: int = 1,
                       head_layers: int = 1,
                       head_norm: bool = False,
                       use_motion: bool = False,
                       motion_window: int = 1,
                       random_goal_tiles: bool = False,
                       soft_q: bool = False,
                       soft_alpha: float = 0.01,
                       softmax_behavior: bool = False,
                       softmax_behavior_temp: float = 0.1) -> None:
        self.env = env
        # Soft-Q (entropy-regularized) value backup + softmax behavior policy.
        # Hard-Q greedy is a deterministic map over a deterministic env, so it can
        # settle into structural limit cycles (A->left->A'->right->A). Soft-Q makes
        # the policy stochastic by construction and the target a soft value
        # alpha*logsumexp(Q/alpha); alpha is the (fixed) temperature — the one knob.
        # Eval a soft-Q model with softmax at temp=alpha (matched by construction).
        self.soft_q = soft_q
        self.soft_alpha = soft_alpha
        # Hard-Q behavior policy = softmax_rel (the deploy readout), NOT argmax.
        # The greedy map over a deterministic env settles into limit cycles, and
        # the cure that works at deploy is softmax_rel sampling (scale-invariant
        # temperature). But the champion trains epsilon-greedy and only deploys
        # softmax_rel, so the Q-function never collected data under the policy we
        # ship. This collects the exploit steps under that same distribution to
        # close the train/deploy mismatch. Unlike soft_q this does NOT touch the
        # Bellman target (no entropy term, no value flattening) — it only changes
        # action selection during rollout.
        self.softmax_behavior = softmax_behavior
        self.softmax_behavior_temp = softmax_behavior_temp
        # When True, each episode's goal is a uniformly-sampled valid floor tile
        # (whole-map coverage) instead of the env's trash/fixture goal — trains a
        # navigator that reaches arbitrary commanded coords, not just trash spots.
        self.random_goal_tiles = random_goal_tiles
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        raw_obs, _ = self.env.reset()
        obs = self.process_observation(raw_obs["observation"])

        self.n_actions = self.env.action_space.n  # type: ignore[union-attr]

        # Coordinate reframing: goal is [robot_x, robot_y, goal_x, goal_y].
        self.goal_dim = 4

        # Previous-motion input (anti-oscillation): last action + velocity, so the
        # net can tell "approaching" from "stuck/reversing" (position alone can't).
        # motion_window>1 adds a windowed net-displacement term so the *moving*
        # limit cycle (spinning) becomes observable, not just the stationary stick.
        self.use_motion = use_motion
        self.motion_window = motion_window
        self.motion_dim = motion_dim(self.n_actions, motion_window) if use_motion else 0

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device=self.device,
            output_device=self.device,
            goal_dim=self.goal_dim,
            motion_dim=self.motion_dim,
        )

        self.q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
            goal_dim=self.goal_dim,
            goal_layers=goal_layers,
            head_layers=head_layers,
            head_norm=head_norm,
            use_motion=use_motion,
            motion_in_dim=self.motion_dim or None,
            motion_window=motion_window,
        ).to(self.device)

        self.target_q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
            goal_dim=self.goal_dim,
            goal_layers=goal_layers,
            head_layers=head_layers,
            head_norm=head_norm,
            use_motion=use_motion,
            motion_in_dim=self.motion_dim or None,
            motion_window=motion_window,
        ).to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.00005)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.977

        self.target_update_interval = target_update_interval
        self.total_steps = 0          # grad-step counter
        self.total_env_steps = 0      # env-step counter (for UTD visibility)
        self.episode_buffer = EpisodeBuffer()

        # UTD: gradient updates per ENV step (replaces the old fixed 800-per-
        # episode block, which silently raised UTD as episodes shortened).
        self.updates_per_step = 1

        self.best_chain_score = -1.0

        # Lazily-built non-goal env for the chained task-score metric (the real
        # deployment metric: trash>>fridge>>human>>door>>human, score out of N).
        self._chain_env = None

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    def _reset_goal(self, base, desired_goal):
        """If random-tile goals are on, replace the env's desired_goal with a
        uniformly-sampled valid floor tile and sync the env's internal goal so
        reward/termination track it. Covers the whole map (incl. doorway/fixture
        coords that trash never spawns near). Returns the goal to use.

        The goal is not rendered into the observation (egocentric viewport), so
        overriding it post-reset is safe — only the goal vector changes.
        """
        if not self.random_goal_tiles:
            return desired_goal
        tiles = base._map.valid_floor_tiles()
        col, row = tiles[int(base.np_random.integers(0, len(tiles)))]
        gx, gy = base._map.tile_to_pixel(col, row)
        goal = np.array([float(gx), float(gy)], dtype=np.float32)
        base._desired_goal = goal
        return goal

    def _q_forward(self, model, obs, goal, motion):
        """Single-sample greedy forward; builds the motion tensor when enabled."""
        obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        motion_t = None
        if self.use_motion:
            motion_t = torch.as_tensor(motion, dtype=torch.float32, device=self.device).unsqueeze(0)
        return model(obs_t, goal_t, motion_t)

    def select_action(self, obs, goal, motion=None):
        """goal: absolute coords [robot_x, robot_y, goal_x, goal_y] (map pixels).
        motion: previous-motion feature (used only when use_motion)."""
        # Soft-Q: sample a ~ softmax(Q/alpha). The entropy temperature IS the
        # exploration mechanism, so no epsilon schedule (early flat Q -> ~uniform).
        if self.soft_q:
            with torch.no_grad():
                q = self._q_forward(self.q_model, obs, goal, motion).squeeze(0)
                probs = F.softmax(q / self.soft_alpha, dim=0)
                return int(torch.multinomial(probs, 1).item())
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q = self._q_forward(self.q_model, obs, goal, motion).squeeze(0)
            if self.softmax_behavior:
                # softmax_rel: scale-invariant temperature = temp * per-state Q
                # spread, identical to evaluate/spin_metric/chained_eval. Trains
                # the exploit steps under the deployed policy. epsilon is kept for
                # early uniform coverage (decays 1.0 -> 0.1); softmax_rel alone
                # would sharpen on noise when Q is still flat.
                probs = softmax_rel_probs(q, self.softmax_behavior_temp)
                return int(torch.multinomial(probs, 1).item())
            return int(q.argmax().item())

    def train_step(self, batch_size):
        (obs, actions, rewards, next_obs, dones, goals, next_goals,
         motions, next_motions) = self.memory.sample_buffer(batch_size)

        obs      = obs      / 255.0
        next_obs = next_obs / 255.0

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones   = dones.unsqueeze(1).float()

        q_sa = self.q_model(obs, goals, motions).gather(1, actions)

        with torch.no_grad():
            next_q_all = self.target_q_model(next_obs, next_goals, next_motions)
            if self.soft_q:
                # Soft value: V(s') = alpha * logsumexp(Q_target(s',.)/alpha).
                next_v = self.soft_alpha * torch.logsumexp(
                    next_q_all / self.soft_alpha, dim=1, keepdim=True)
            else:
                # Double-DQN: action from online net, value from target net.
                next_actions = self.q_model(next_obs, next_goals, next_motions).argmax(dim=1, keepdim=True)
                next_v = next_q_all.gather(1, next_actions)
            targets = rewards + (1 - dones) * self.gamma * next_v

        loss = F.smooth_l1_loss(q_sa, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.total_steps += 1
        return loss.item()

    def save(self):
        self.q_model.save_the_model("q_model", verbose=True)

    def load(self):
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    def save_best(self, episode, chain_score):
        path = "checkpoints/q_model_best.pt"
        torch.save({
            "q_model": self.q_model.state_dict(),
            "episode": episode,
            "chain_score": chain_score,
        }, path)
        print(f"  New best checkpoint saved (episode={episode}, chain_score={chain_score:.2f})")

    def greedy_eval(self, n_episodes: int = 20) -> float:
        """Greedy single-goal eval mirroring evaluate.py (full episode, reward>0.5),
        so numbers are comparable to the proven baseline. Random spawn each episode.
        Returns the reach rate; chain_eval is the deployment metric."""
        self.q_model.eval()
        successes = 0

        for _ in range(n_episodes):
            raw_obs, _   = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot
            desired_goal = self._reset_goal(base, desired_goal)
            ms           = MotionState(self.n_actions, self.motion_window)

            done = False
            ep_reward = 0.0
            while not done:
                goal_vec = world_coords(r.x, r.y,
                                        desired_goal[0], desired_goal[1])
                motion = ms.vec(r.x, r.y)
                with torch.no_grad():
                    action = self._q_forward(self.q_model, obs, goal_vec, motion).argmax(dim=1).item()
                ms.commit(r.x, r.y, action)

                raw_next, reward, term, trunc, _ = self.env.step(action)
                obs = self.process_observation(raw_next["observation"])
                ep_reward += float(reward)
                done = term or trunc

            if ep_reward > 0.5:
                successes += 1

        self.q_model.train()
        return successes / n_episodes

    def chain_eval(self, n_episodes: int = 5):
        """The real deployment metric: run the static task chain in the non-goal
        env (all task items loaded), greedy, pose persisting leg-to-leg. Returns
        (mean_score, full_chain_rate) where score is legs reached out of
        len(DEFAULT_CHAIN). Reuses the offline run_chain so the TB number equals
        what chained_eval.py reports. Fixed seeds -> low-noise curve.

        Also returns mean_spin: the spin fraction (scripts/spin_metric.py, shared
        spin_fraction) averaged over every leg -- the metric this experiment
        actually targets, so it's watchable live instead of only post-hoc."""
        if self._chain_env is None:
            self._chain_env = gym.make(
                "HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                map_name="default", random_start=True,
            )
        self.q_model.eval()
        n_legs = len(DEFAULT_CHAIN)
        # A soft-Q model is meant to be used stochastically; score it the way it
        # deploys (softmax at temp=alpha) so the metric isn't a greedy undersell
        # and is comparable to the hard-Q champion's softmax readout.
        readout = "softmax" if self.soft_q else "greedy"
        temp = self.soft_alpha if self.soft_q else 0.01
        move_min, net_max = spin_thresholds(SPIN_WINDOW)
        total, full = 0, 0
        spins = []
        for i in range(n_episodes):
            legs = run_chain(self.q_model, self._chain_env, DEFAULT_CHAIN,
                             self.device, readout, temp, seed=i)
            reached = sum(1 for _, r, *_ in legs if r)
            total += reached
            if reached == n_legs:
                full += 1
            spins.extend(spin_fraction(pos, SPIN_WINDOW, move_min, net_max)
                         for _, _, _, pos in legs)
        self.q_model.train()
        mean_spin = sum(spins) / len(spins) if spins else 0.0
        return total / n_episodes, full / n_episodes, mean_spin

    def train(self, episodes=1000, batch_size=64, run_tag=None,
              eval_interval=50, eval_episodes=20, chain_eval_interval=10,
              her_anneal_start=None,
              reach_start=None, reach_end=None,
              reach_anneal_start=0, reach_anneal_end=None):
        # Success-radius curriculum: when reach_start is set, the per-episode reach/
        # terminal radius anneals reach_start -> reach_end over [anneal_start,
        # anneal_end], replacing the env's fixed 79px reward+termination. Pure HER
        # artifact -- no env change (see goal_geometry.reach_reward). reach_start
        # None preserves the original env-reward behavior exactly.
        use_reach_curriculum = reach_start is not None
        if use_reach_curriculum and reach_anneal_end is None:
            reach_anneal_end = episodes
        if run_tag is None:
            try:
                refs = subprocess.check_output(
                    ['git', 'for-each-ref', '--format=%(refname:short)',
                     '--points-at', 'HEAD', 'refs/remotes/origin/'],
                    stderr=subprocess.DEVNULL).decode().strip()
                if refs:
                    run_tag = refs.splitlines()[0].replace('origin/', '')
                if not run_tag:
                    run_tag = subprocess.check_output(
                        ['git', 'branch', '--show-current'],
                        stderr=subprocess.DEVNULL).decode().strip()
                if not run_tag:
                    run_tag = 'unknown'
            except Exception:
                run_tag = 'unknown'

        writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        for episode in range(episodes):
            raw_obs, _   = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot
            desired_goal = self._reset_goal(base, desired_goal)
            ms           = MotionState(self.n_actions, self.motion_window)

            if use_reach_curriculum:
                reach_radius = reach_radius_at(episode, reach_start, reach_end,
                                               reach_anneal_start, reach_anneal_end)
                writer.add_scalar('Train/reach_radius', reach_radius, episode)

            done = False
            episode_reward = 0.0
            episode_loss   = 0.0
            episode_steps  = 0

            while not done:
                heading_prev = r.angle
                pos_prev     = np.array([r.x, r.y], dtype=np.float32)
                goal_vec     = world_coords(r.x, r.y,
                                            desired_goal[0], desired_goal[1])
                motion_prev  = ms.vec(r.x, r.y)
                action       = self.select_action(obs, goal_vec, motion_prev)
                ms.commit(r.x, r.y, action)

                raw_next, env_reward, env_term, trunc, _ = self.env.step(action)
                next_obs     = self.process_observation(raw_next["observation"])
                heading_next = r.angle
                pos_next     = np.array([r.x, r.y], dtype=np.float32)
                # motion at s' == velocity into s' + windowed net from history;
                # ms.vec at the new pose reproduces it (history already holds the
                # pose we just committed), so the windowed term comes for free.
                motion_next  = ms.vec(pos_next[0], pos_next[1])
                if use_reach_curriculum:
                    # Recompute reward + terminal at the scheduled radius from the
                    # pose we already have; the env's fixed-79px reward/termination
                    # is ignored. Driving the rollout's `done` off this radius means
                    # the robot keeps approaching past 79px -- gathering on-policy
                    # data in the dead zone that the env would otherwise cut short.
                    reward = float(reach_reward(pos_next, desired_goal, reach_radius))
                    term   = reward > 0.5
                else:
                    reward, term = env_reward, env_term
                done = term or trunc

                # Store term (not trunc): a timeout is not a terminal state, so the
                # target should still bootstrap from next_obs. Storing trunc as done
                # trains Q toward 0 at far-from-goal states.
                self.episode_buffer.store(
                    obs, action, reward, next_obs, term,
                    achieved_prev=pos_prev,
                    achieved_next=pos_next,
                    heading_prev=heading_prev,
                    heading_next=heading_next,
                    motion_prev=motion_prev,
                    motion_next=motion_next,
                )
                episode_reward += float(reward)
                episode_steps  += 1
                self.total_env_steps += 1
                obs = next_obs

                # UTD: fixed gradient updates per ENV step (not per episode), so
                # the update-to-data ratio stays constant as episodes shorten.
                for _ in range(self.updates_per_step):
                    if self.memory.can_sample(batch_size):
                        episode_loss += self.train_step(batch_size)

            # HER curriculum: bootstrap at full K, then anneal hindsight 2 -> 0
            # after her_anneal_start so the buffer's marginal inflow leans toward
            # real (un-relabeled) data — like a fine-tune off the relabeled diet.
            k_eff = self.episode_buffer.K
            if her_anneal_start is not None and episode >= her_anneal_start:
                span = max(1, episodes - her_anneal_start)
                frac = min(1.0, (episode - her_anneal_start) / span)
                k_eff = self.episode_buffer.K * (1.0 - frac)
            # HER reward: curriculum radius when enabled (so relabeled goals score at
            # the same tightening radius as the rollout), else the env's fixed bar.
            her_reward = (
                (lambda a, d, info: reach_reward(a, d, reach_radius))
                if use_reach_curriculum
                else self.env.unwrapped.compute_reward)  # type: ignore[attr-defined]
            self.episode_buffer.send_to(
                self.memory,
                desired_goal=desired_goal,
                compute_reward=her_reward,
                k=k_eff,
            )
            self.episode_buffer.clear()
            writer.add_scalar("Train/hindsight_k", k_eff, episode)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon",         self.epsilon,   episode)
            writer.add_scalar("Train/avg_q_loss",      avg_loss,       episode)
            writer.add_scalar("Train/episode_steps",   episode_steps,  episode)
            writer.add_scalar("Train/total_env_steps",  self.total_env_steps, episode)
            writer.add_scalar("Buffer/fill", min(self.memory.mem_ctr, self.memory.mem_size), episode)

            if episode % 10 == 0:
                self.save()

            # Honest greedy eval — single-goal nav health (the chain eval below is
            # the deployment metric). Directness now lives in chain_spin_fraction.
            if episode % eval_interval == 0:
                reach_rate = self.greedy_eval(n_episodes=eval_episodes)
                writer.add_scalar("Eval/greedy_reach_rate", reach_rate, episode)
                print(f"  [Eval] episode {episode}: greedy reach_rate={reach_rate:.3f}")

            # Chained task score — the real deployment metric, logged often, and
            # the criterion for the "best" checkpoint (greedy reach_rate selected
            # the wrong models: it's a weaker readout than what we deploy on).
            if episode % chain_eval_interval == 0:
                chain_score, chain_full, chain_spin = self.chain_eval()
                writer.add_scalar("Eval/chain_score", chain_score, episode)
                writer.add_scalar("Eval/chain_full", chain_full, episode)
                writer.add_scalar("Eval/chain_spin_fraction", chain_spin, episode)
                print(f"  [Chain] episode {episode}: score={chain_score:.2f}/"
                      f"{len(DEFAULT_CHAIN)} | full_chain={chain_full:.2f} | "
                      f"spin={chain_spin:.3f}")

                if chain_score > self.best_chain_score:
                    self.best_chain_score = chain_score
                    self.save_best(episode, chain_score)

    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []

        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            pos          = raw_obs["achieved_goal"]
            done = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
                    goal_vec = world_coords(pos[0], pos[1],
                                            desired_goal[0], desired_goal[1])
                    goal_t = torch.as_tensor(goal_vec, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                    action = self.q_model(obs_t, goal_t).argmax(dim=1).item()
                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(raw_next["observation"])
                done = term or trunc
                episode_reward += float(reward)
                obs = next_obs
                pos = raw_next["achieved_goal"]

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards
