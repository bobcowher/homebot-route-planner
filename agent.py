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
from goal_geometry import world_coords
from models.q_model import QModel
from torch.utils.tensorboard.writer import SummaryWriter


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 100000,
                       target_update_interval: int = 1000,
                       goal_layers: int = 1,
                       head_layers: int = 1,
                       random_goal_tiles: bool = False) -> None:
        self.env = env
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

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device=self.device,
            output_device=self.device,
            goal_dim=self.goal_dim,
        )

        self.q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
            goal_dim=self.goal_dim,
            goal_layers=goal_layers,
            head_layers=head_layers,
        ).to(self.device)

        self.target_q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
            goal_dim=self.goal_dim,
            goal_layers=goal_layers,
            head_layers=head_layers,
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

        self.best_reach_rate = -1.0

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

    def select_action(self, obs, goal):
        """goal: absolute coords [robot_x, robot_y, goal_x, goal_y] (map pixels)."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_model(obs_t, goal_t).argmax(dim=1).item()

    def train_step(self, batch_size):
        obs, actions, rewards, next_obs, dones, goals, next_goals = self.memory.sample_buffer(batch_size)

        obs      = obs      / 255.0
        next_obs = next_obs / 255.0

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones   = dones.unsqueeze(1).float()

        q_sa = self.q_model(obs, goals).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_model(next_obs, next_goals).argmax(dim=1, keepdim=True)
            next_q       = self.target_q_model(next_obs, next_goals).gather(1, next_actions)
            targets      = rewards + (1 - dones) * self.gamma * next_q

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

    def save_best(self, episode, reach_rate):
        path = "checkpoints/q_model_best.pt"
        torch.save({
            "q_model": self.q_model.state_dict(),
            "episode": episode,
            "reach_rate": reach_rate,
        }, path)
        print(f"  New best checkpoint saved (episode={episode}, reach_rate={reach_rate:.3f})")

    def greedy_eval(self, n_episodes: int = 20) -> float:
        """Greedy eval mirroring her/evaluate.py (full episode, reward>0.5), so
        numbers are comparable to the proven baseline. Rung 1 keeps the random
        spawn (the only difference from Rung 0's eval).

        self.last_avg_success_steps tracks steps on successful episodes — the
        circling diagnostic.
        """
        self.q_model.eval()
        successes = 0
        success_steps = []

        for _ in range(n_episodes):
            raw_obs, _   = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot
            desired_goal = self._reset_goal(base, desired_goal)

            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                goal_vec = world_coords(r.x, r.y,
                                        desired_goal[0], desired_goal[1])
                with torch.no_grad():
                    obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
                    goal_t = torch.as_tensor(goal_vec, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                    action = self.q_model(obs_t, goal_t).argmax(dim=1).item()

                raw_next, reward, term, trunc, _ = self.env.step(action)
                obs = self.process_observation(raw_next["observation"])
                ep_reward += float(reward)
                steps += 1
                done = term or trunc

            if ep_reward > 0.5:
                successes += 1
                success_steps.append(steps)

        self.q_model.train()
        self.last_avg_success_steps = (
            sum(success_steps) / len(success_steps) if success_steps else 0.0
        )
        return successes / n_episodes

    def softmax_eval(self, n_episodes: int = 20, temp: float = 0.1):
        """Same as greedy_eval but samples a ~ softmax(Q(s,.)/temp) instead of
        argmax. Tests whether stochastic action selection breaks the limit cycles
        that pin greedy reach far below training reward (the 0.9-train/0.25-greedy
        gap). Returns (reach_rate, avg_success_steps)."""
        self.q_model.eval()
        successes = 0
        success_steps = []

        for _ in range(n_episodes):
            raw_obs, _   = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot
            desired_goal = self._reset_goal(base, desired_goal)

            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                goal_vec = world_coords(r.x, r.y,
                                        desired_goal[0], desired_goal[1])
                with torch.no_grad():
                    obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
                    goal_t = torch.as_tensor(goal_vec, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                    q      = self.q_model(obs_t, goal_t).squeeze(0)
                    probs  = F.softmax(q / temp, dim=0)
                    action = int(torch.multinomial(probs, 1).item())

                raw_next, reward, term, trunc, _ = self.env.step(action)
                obs = self.process_observation(raw_next["observation"])
                ep_reward += float(reward)
                steps += 1
                done = term or trunc

            if ep_reward > 0.5:
                successes += 1
                success_steps.append(steps)

        self.q_model.train()
        avg_steps = sum(success_steps) / len(success_steps) if success_steps else 0.0
        return successes / n_episodes, avg_steps

    def train(self, episodes=1000, batch_size=64, run_tag=None,
              eval_interval=50, eval_episodes=20):
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

            done = False
            episode_reward = 0.0
            episode_loss   = 0.0
            episode_steps  = 0

            while not done:
                heading_prev = r.angle
                pos_prev     = np.array([r.x, r.y], dtype=np.float32)
                goal_vec     = world_coords(r.x, r.y,
                                            desired_goal[0], desired_goal[1])
                action       = self.select_action(obs, goal_vec)

                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs     = self.process_observation(raw_next["observation"])
                heading_next = r.angle
                pos_next     = np.array([r.x, r.y], dtype=np.float32)
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

            self.episode_buffer.send_to(
                self.memory,
                desired_goal=desired_goal,
                compute_reward=self.env.unwrapped.compute_reward,  # type: ignore[attr-defined]
            )
            self.episode_buffer.clear()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon",         self.epsilon,   episode)
            writer.add_scalar("Train/avg_q_loss",      avg_loss,       episode)
            writer.add_scalar("Train/episode_steps",   episode_steps,  episode)
            writer.add_scalar("Train/total_grad_steps", self.total_steps,     episode)
            writer.add_scalar("Train/total_env_steps",  self.total_env_steps, episode)
            writer.add_scalar("Train/updates_per_env_step", self.updates_per_step, episode)
            writer.add_scalar("Buffer/fill", min(self.memory.mem_ctr, self.memory.mem_size), episode)

            if episode % 10 == 0:
                self.save()

            # Honest greedy eval — the real metric for the ablation ladder.
            if episode % eval_interval == 0:
                reach_rate = self.greedy_eval(n_episodes=eval_episodes)
                writer.add_scalar("Eval/greedy_reach_rate", reach_rate, episode)
                writer.add_scalar("Eval/avg_success_steps",
                                  getattr(self, "last_avg_success_steps", 0.0), episode)
                print(f"  [Eval] episode {episode}: greedy reach_rate={reach_rate:.3f} "
                      f"| avg_success_steps={getattr(self, 'last_avg_success_steps', 0.0):.0f}")

                # Softmax (Boltzmann) eval readout: does sampling break the greedy
                # limit cycles and close the train/greedy gap? Sweep two temps.
                for temp in (0.01, 0.025, 0.05):
                    sm_reach, sm_steps = self.softmax_eval(n_episodes=eval_episodes, temp=temp)
                    writer.add_scalar(f"Eval/softmax_reach_rate_t{temp}", sm_reach, episode)
                    writer.add_scalar(f"Eval/softmax_avg_success_steps_t{temp}", sm_steps, episode)
                    print(f"  [Eval] episode {episode}: softmax(t={temp}) "
                          f"reach_rate={sm_reach:.3f} | avg_success_steps={sm_steps:.0f}")

                if reach_rate > self.best_reach_rate:
                    self.best_reach_rate = reach_rate
                    self.save_best(episode, reach_rate)

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
