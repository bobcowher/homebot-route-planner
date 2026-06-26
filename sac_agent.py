"""Discrete SAC agent for HomeBotGoalEnv (discrete action mode).

Observation pipeline matches the DQN champion:
  image  : 96×96 RGB → resize → permute(2,0,1) → uint8 tensor
  goal   : noisy_world_vector(rx, ry, gx, gy, noise_std) → [dx, dy] float32
  motion : MotionStateDiscrete → [dx/step, dy/step, 0, 0] float32

Discrete SAC update (Christodoulou 2019):
  Critic target: V(s') = Σ_a π(a|s')[Q_target(s',a) - α log π(a|s')]
  Critic loss:   MSE(Q(s, a_taken), r + γ V(s'))
  Actor loss:    Σ_a π(a|s)[α log π(a|s) - min_Q(s,a)]
No reparameterisation needed — expectation computed exactly over discrete actions.
"""
import cv2
import datetime
import math
import os
import subprocess
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from goal_geometry import noisy_world_vector, reach_reward, reach_radius_at
from sac_buffer import SACReplayBuffer
from sac_episode_buffer import SACEpisodeBuffer
from sac_model import DiscreteQNet, DiscretePolicy
from sac_motion import MotionStateDiscrete


class SACAgent:
    def __init__(self, env, max_buffer_size=200000,
                 gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4,
                 goal_noise_std=30.0,
                 autotune_alpha=True, target_entropy_ratio=0.7,
                 alpha_lr=1e-4, alpha_min=0.05, alpha_max=0.3,
                 actor_head_layers=4, actor_head_hidden=512,
                 critic_head_layers=4, critic_head_hidden=512):
        self.env = env
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # current temperature (float); updated each step when autotuning
        self._alpha_init = alpha  # starting temperature for the alpha-decay schedule
        self.goal_noise_std = goal_noise_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Automatic entropy tuning. Fixed alpha cannot hold a categorical policy
        # open on sparse reward — it collapses to a deterministic non-solution
        # (entropy -> 0) before it ever finds the goal. We instead learn alpha to
        # hold the policy's entropy at a target: target = ratio * log(n_actions),
        # ratio < 1 so the converged policy can still sharpen toward the goal.
        #
        # alpha_min FLOOR: during the rewardless cold-start the critic is flat, so
        # the policy stays ~uniform (entropy ~= max) no matter what alpha does —
        # nothing to sharpen toward yet. The tuner reads entropy > target and
        # drives alpha toward 0 (~1000 updates/episode), and once HER finally gives
        # the critic signal the now-dead alpha lets the policy collapse unregularised
        # (this is run-335's alpha 0.074 -> 0.003 in 11 episodes). Flooring alpha keeps
        # an entropy floor through the cold-start; the tuner still raises it above the
        # floor once real Q-spread appears. alpha_lr is also decoupled (gentler) so the
        # temperature doesn't overshoot across the per-episode update burst.
        #
        # alpha_max CEILING: these episodes never terminate (the agent doesn't reach
        # the goal -> 1000-step timeout -> mask=1 always), so the soft value adds an
        # entropy bonus alpha*H at every bootstrap step that accumulates over the full
        # horizon: mean_q ~= alpha*H/(1-gamma). Unbounded alpha makes Q (and the tuner)
        # run away: Q inflates -> policy sharpens -> entropy < target -> tuner raises
        # alpha -> bigger bonus -> Q inflates more (run-336: alpha 0.05 -> 1.24, mean_q
        # -> 176, critic_loss -> 5090). Capping alpha bounds the bonus (~alpha_max*H/(1-g))
        # so Q converges. The band [min, max] gives the tuner authority to fight collapse
        # (max 0.3 >> the fixed 0.1 that collapsed run 334) without letting it diverge.
        self.autotune_alpha = autotune_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_entropy = target_entropy_ratio * math.log(self.n_actions)
        self.log_alpha = torch.tensor(
            math.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.log_alpha_min = math.log(alpha_min)
        self.log_alpha_max = math.log(alpha_max)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        # Asymmetric heads: actor deep (compositional policy, like the Q-champion's
        # head_layers=4), critic optionally flat & wide (value regression — depth amplifies
        # the bootstrap overestimation that diverged the critic; wide-shallow is steadier
        # and gives cleaner Q-spread for the actor to exploit). Defaults = symmetric 4x512.
        self.critic = DiscreteQNet(self.n_actions, name="sac_critic",
                                   head_layers=critic_head_layers, head_hidden=critic_head_hidden).to(self.device)
        self.critic_target = DiscreteQNet(self.n_actions, name="sac_critic_target",
                                          head_layers=critic_head_layers, head_hidden=critic_head_hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.policy = DiscretePolicy(self.n_actions, name="sac_policy",
                                     head_layers=actor_head_layers, head_hidden=actor_head_hidden).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)

        self.memory = SACReplayBuffer(max_buffer_size, device=str(self.device))
        self.episode_buffer = SACEpisodeBuffer()

        self.total_env_steps = 0
        self.total_grad_steps = 0

    # ------------------------------------------------------------------
    # Observation pipeline (mirrors DQN agent.process_observation)
    # ------------------------------------------------------------------

    def process_observation(self, obs_hwc):
        obs_hwc = cv2.resize(obs_hwc, (96, 96), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(obs_hwc).permute(2, 0, 1)

    def _to_device_float(self, img_tensor):
        return img_tensor.unsqueeze(0).float().to(self.device) / 255.0

    def _goal_tensor(self, goal_np):
        return torch.as_tensor(goal_np, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _motion_tensor(self, motion_np):
        return torch.as_tensor(motion_np, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs_tensor, goal_np, motion_np, evaluate=False):
        img  = self._to_device_float(obs_tensor)
        goal = self._goal_tensor(goal_np)
        mot  = self._motion_tensor(motion_np)
        action, _ = self.policy.get_action(img, goal, mot, evaluate=evaluate)
        return action.item()

    # ------------------------------------------------------------------
    # Discrete SAC update
    # ------------------------------------------------------------------

    def update_parameters(self, batch_size):
        imgs, goals, motions, actions, rewards, \
        next_imgs, next_goals, next_motions, dones = self.memory.sample_buffer(batch_size)

        rewards = rewards.unsqueeze(1)
        mask    = (~dones).float().unsqueeze(1)

        # ---- Critic target: soft value V(s') -------------------------
        with torch.no_grad():
            next_probs, next_log_probs = self.policy(next_imgs, next_goals, next_motions)
            q1_next, q2_next = self.critic_target(next_imgs, next_goals, next_motions)
            min_q_next = torch.min(q1_next, q2_next)          # (B, n_actions)
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + mask * self.gamma * v_next   # (B, 1)

        # ---- Critic loss: Q for taken action vs target ---------------
        q1, q2 = self.critic(imgs, goals, motions)
        q1_a = q1.gather(1, actions.unsqueeze(1))             # (B, 1)
        q2_a = q2.gather(1, actions.unsqueeze(1))
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor loss: exact expectation over actions ---------------
        probs, log_probs = self.policy(imgs, goals, motions)
        with torch.no_grad():
            q1, q2 = self.critic(imgs, goals, motions)
        min_q = torch.min(q1, q2)
        # L = Σ_a π(a|s)[α log π(a|s) - min_Q(s,a)]
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # ---- Temperature: pull policy entropy toward target ----------
        # alpha loss = E_a~π[ -log_alpha * (log π(a|s) + H_target) ], detached
        # from the policy so only log_alpha moves. When entropy < target the
        # bracket is positive -> alpha rises (more exploration), and vice versa.
        if self.autotune_alpha:
            alpha_loss = (probs.detach() *
                          (-self.log_alpha * (log_probs.detach() + self.target_entropy))
                          ).sum(dim=1).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.log_alpha_min, max=self.log_alpha_max)
            self.alpha = self.log_alpha.exp().item()

        # ---- Polyak target update ------------------------------------
        for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

        self.total_grad_steps += 1
        mean_q   = min_q.mean().item()
        entropy  = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean().item()
        return critic_loss.item(), actor_loss.item(), mean_q, entropy

    # ------------------------------------------------------------------
    # Run-tag detection
    # ------------------------------------------------------------------

    def _run_tag(self):
        try:
            refs = subprocess.check_output(
                ['git', 'for-each-ref', '--format=%(refname:short)',
                 '--points-at', 'HEAD', 'refs/remotes/origin/'],
                stderr=subprocess.DEVNULL).decode().strip()
            tag = refs.splitlines()[0].replace('origin/', '') if refs else None
            if not tag:
                tag = subprocess.check_output(
                    ['git', 'branch', '--show-current'],
                    stderr=subprocess.DEVNULL).decode().strip()
            return tag or 'unknown'
        except Exception:
            return 'unknown'

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _spawn_near_goal(self, base, start_dist):
        """Start-distance curriculum: reposition the robot onto a valid floor tile in
        [start_dist_min, start_dist] px of the goal, then rebuild the obs. The discrete
        policy moves 4px/step, so a random walk only diffuses ~4*sqrt(T) ~ 126px over a
        1000-step episode — far spawns are physically unreachable by exploration and
        yield no learning signal. Spawning near the goal makes navigation short enough
        to actually reach (real reward + HER on directed trajectories); start_dist then
        expands toward the full map. start_dist_min sits just outside the 79px reach
        radius so the episode still requires real navigation, not a step-0 win."""
        gx, gy = float(base._desired_goal[0]), float(base._desired_goal[1])
        ts = base._map.tile_size
        cands = []
        for col, row in base._map.valid_floor_tiles():
            px, py = base._map.tile_to_pixel(col, row)
            d = math.hypot(px - gx, py - gy)
            if self._start_dist_min <= d <= start_dist and not base._robot._collides(
                    float(px), float(py), base._map.wall_solid, ts,
                    base._map.fixture_pixel_rects):
                cands.append((float(px), float(py)))
        if cands:  # else: keep the env's own spawn (e.g. goal cornered, no tile in band)
            px, py = cands[int(base.np_random.integers(0, len(cands)))]
            base._robot.x, base._robot.y = px, py
        return base._build_obs()

    def _run_episode(self, collect_only=False, batch_size=64, reach_radius=None,
                     start_dist=None):
        # reach_radius (success-radius curriculum): when set, the rollout reward and
        # terminal are recomputed at this radius from the robot pose, overriding the
        # env's fixed 79px. reach_radius=None preserves env behavior.
        # start_dist (start-distance curriculum): when set, the robot is respawned
        # within start_dist of the goal (see _spawn_near_goal) — the exploration fix.
        use_curriculum = reach_radius is not None
        raw_obs, _ = self.env.reset()
        base = self.env.unwrapped
        if start_dist is not None:
            raw_obs = self._spawn_near_goal(base, start_dist)
        r = base._robot
        desired_goal = raw_obs["desired_goal"]
        ms  = MotionStateDiscrete()
        obs = self.process_observation(raw_obs["observation"])

        done = False
        episode_reward = 0.0
        episode_steps  = 0
        critic_loss_sum = actor_loss_sum = mean_q_sum = entropy_sum = 0.0
        update_count = 0

        while not done:
            pos_prev    = np.array([r.x, r.y], dtype=np.float32)
            motion_prev = ms.vec(r.x, r.y)
            goal_prev   = noisy_world_vector(r.x, r.y, desired_goal[0], desired_goal[1],
                                             self.goal_noise_std)

            if collect_only:
                action = int(self.env.action_space.sample())
            else:
                action = self.select_action(obs, goal_prev, motion_prev)

            ms.commit(r.x, r.y, action)
            raw_next, env_reward, env_term, trunc, _ = self.env.step(action)
            next_obs = self.process_observation(raw_next["observation"])

            pos_next    = np.array([r.x, r.y], dtype=np.float32)
            motion_next = ms.vec(pos_next[0], pos_next[1])
            if use_curriculum:
                reward = float(reach_reward(pos_next, desired_goal, reach_radius))
                term   = reward > 0.5
            else:
                reward, term = env_reward, env_term
            done = term or trunc
            self.total_env_steps += 1

            self.episode_buffer.store(
                obs, next_obs, action, reward, term,
                achieved_prev=pos_prev, achieved_next=pos_next,
                motion_prev=motion_prev, motion_next=motion_next,
            )
            episode_reward += float(reward)
            episode_steps  += 1
            obs = next_obs

            if not collect_only and self.memory.can_sample(batch_size):
                cl, al, mq, ent = self.update_parameters(batch_size)
                critic_loss_sum += cl
                actor_loss_sum  += al
                mean_q_sum      += mq
                entropy_sum     += ent
                update_count    += 1

        # HER reward: at the curriculum radius when active (so relabeled goals score
        # at the same radius the rollout terminated on), else the env's fixed bar.
        her_reward = (
            (lambda a, d, info: reach_reward(a, d, reach_radius))
            if use_curriculum else base.compute_reward
        )
        self.episode_buffer.send_to(
            self.memory, desired_goal=desired_goal,
            compute_reward=her_reward,
            goal_noise_std=self.goal_noise_std,
        )
        self.episode_buffer.clear()
        return episode_reward, episode_steps, critic_loss_sum, actor_loss_sum, mean_q_sum, entropy_sum, update_count

    def train(self, episodes=900, batch_size=64, run_tag=None, warmup_steps=5000,
              reach_start=None, reach_end=None,
              reach_anneal_start=0, reach_anneal_end=None,
              start_dist_start=None, start_dist_max=900.0, start_dist_step=15.0,
              start_dist_window=25, start_dist_threshold=0.6, start_dist_min=90.0,
              alpha_anneal_to=None, alpha_anneal_episodes=None):
        # Alpha (temperature) decay schedule — the epsilon-greedy analog. When
        # alpha_anneal_to is set (and autotune is off), alpha decays geometrically from
        # its init value to alpha_anneal_to over alpha_anneal_episodes, then holds. This
        # is the explore->exploit transition: high alpha early (stochastic exploration to
        # feed HER) -> alpha ~0 late (the actor becomes argmax-Q, exploiting the HER-built
        # critic). Run 351 showed SAC's separate max-entropy actor never exploits the HER
        # critic on its own (entropy pinned at max for 1100 eps, 0 reaches) — DQN gets this
        # for free because behaviour IS argmax-Q. Decaying alpha to ~0 restores that.
        anneal_alpha = alpha_anneal_to is not None
        if anneal_alpha and alpha_anneal_episodes is None:
            alpha_anneal_episodes = episodes
        # Success-radius curriculum (optional): the per-episode reach/terminal radius
        # anneals reach_start -> reach_end over [anneal_start, anneal_end]. reach_start
        # None preserves the env's fixed-79px reward/termination exactly.
        #
        # Start-distance curriculum (ADAPTIVE): spawn distance starts at start_dist_start
        # and only grows (+start_dist_step, capped at start_dist_max) once the agent
        # clears start_dist_threshold reach-rate over the last start_dist_window episodes.
        # A fixed time schedule outran learning (run 339: distance hit ~196px by ep43
        # while the agent still reached only ~1/15) — success-gating guarantees the agent
        # masters each distance before the task expands.
        use_curriculum = reach_start is not None
        if use_curriculum and reach_anneal_end is None:
            reach_anneal_end = episodes
        use_start_curriculum = start_dist_start is not None
        self._start_dist_min = start_dist_min
        cur_start_dist = start_dist_start
        recent_reaches = deque(maxlen=start_dist_window)
        run_tag = run_tag or self._run_tag()
        writer  = SummaryWriter(
            f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        # Warmup spawns near the goal too (at the initial distance) so the random
        # buffer already contains real reaches, not just rewardless far-walks.
        warmup_dist = start_dist_start if use_start_curriculum else None
        warmup_done = 0
        while warmup_done < warmup_steps:
            _, ep_steps, *_ = self._run_episode(collect_only=True, batch_size=batch_size,
                                                start_dist=warmup_dist)
            warmup_done += ep_steps
        if warmup_steps > 0:
            print(f"[warmup] {warmup_done} random steps collected")

        for episode in range(episodes):
            if anneal_alpha:
                frac = min(1.0, episode / max(1, alpha_anneal_episodes))
                self.alpha = self._alpha_init * (alpha_anneal_to / self._alpha_init) ** frac
            reach_radius = (
                reach_radius_at(episode, reach_start, reach_end,
                                reach_anneal_start, reach_anneal_end)
                if use_curriculum else None
            )
            start_dist = cur_start_dist if use_start_curriculum else None
            ep_reward, ep_steps, cl_sum, al_sum, mq_sum, ent_sum, n_updates = \
                self._run_episode(collect_only=False, batch_size=batch_size,
                                  reach_radius=reach_radius, start_dist=start_dist)

            # Adaptive curriculum: expand spawn distance once the agent is reliably
            # reaching at the current distance.
            if use_start_curriculum:
                recent_reaches.append(1.0 if ep_reward > 0.5 else 0.0)
                if (len(recent_reaches) == recent_reaches.maxlen
                        and sum(recent_reaches) / len(recent_reaches) >= start_dist_threshold
                        and cur_start_dist < start_dist_max):
                    cur_start_dist = min(cur_start_dist + start_dist_step, start_dist_max)
                    recent_reaches.clear()  # re-measure reach-rate at the new distance

            writer.add_scalar("Train/episode_reward",  ep_reward, episode)
            writer.add_scalar("Train/episode_steps",   ep_steps,  episode)
            if use_curriculum:
                writer.add_scalar("Train/reach_radius", reach_radius, episode)
            if use_start_curriculum:
                writer.add_scalar("Train/start_dist", start_dist, episode)
            if n_updates > 0:
                writer.add_scalar("loss/critic",           cl_sum  / n_updates, episode)
                writer.add_scalar("loss/actor",            al_sum  / n_updates, episode)
                writer.add_scalar("Train/mean_q",          mq_sum  / n_updates, episode)
                writer.add_scalar("Train/policy_entropy",  ent_sum / n_updates, episode)
                writer.add_scalar("Train/alpha",           self.alpha, episode)

            radius_str = f" | radius: {reach_radius:.0f}" if use_curriculum else ""
            sdist_str  = f" | start_dist: {start_dist:.0f}" if use_start_curriculum else ""
            print(f"Episode {episode} | reward: {ep_reward:.2f} | "
                  f"steps: {ep_steps} | alpha: {self.alpha:.3f}{radius_str}{sdist_str}")

            if episode % 50 == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoint(self, evaluate=False):
        self.policy.load_checkpoint()
        self.critic.load_checkpoint()
        if evaluate:
            self.policy.eval()
            self.critic.eval()
