"""CNN-based SAC agent for HomeBotGoalEnv (continuous action mode).

Observation pipeline matches the DQN champion:
  - Image: 96x96 RGB → resize → permute(2,0,1) → uint8 tensor
  - Goal:  noisy_world_vector(rx, ry, gx, gy, noise_std=30) → [dx, dy] float32
  - Motion: [last_linear, last_angular, dx, dy] float32

SAC engine (select_action, update_parameters) ported from sac-fetch/agent.py;
train() is a new HER training loop wiring sac_episode_buffer + sac_motion onto
HomeBotGoalEnv's continuous action_mode.
"""
import cv2
import datetime
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from goal_geometry import noisy_world_vector
from sac_buffer import SACReplayBuffer
from sac_episode_buffer import SACEpisodeBuffer
from sac_model import Critic, Policy
from sac_motion import MotionStateContinuous


class SACAgent:
    def __init__(self, env, action_dim=2, max_buffer_size=200000,
                 gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4,
                 motion_window=1, goal_noise_std=30.0):
        self.env = env
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.motion_window = motion_window
        self.goal_noise_std = goal_noise_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        self.critic = Critic(action_dim, name="sac_critic").to(self.device)
        self.critic_target = Critic(action_dim, name="sac_critic_target").to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.policy = Policy(action_dim, env.action_space, name="sac_policy").to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.memory = SACReplayBuffer(max_buffer_size, action_dim, device=str(self.device))
        self.episode_buffer = SACEpisodeBuffer()

        self.total_env_steps = 0
        self.total_grad_steps = 0

    # ------------------------------------------------------------------
    # Observation pipeline (mirrors DQN agent.process_observation)
    # ------------------------------------------------------------------

    def process_observation(self, obs_hwc):
        """(H, W, 3) uint8 ndarray → (3, H, W) uint8 torch.Tensor."""
        obs_hwc = cv2.resize(obs_hwc, (96, 96), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(obs_hwc).permute(2, 0, 1)

    def _to_device_float(self, img_tensor):
        """(3,96,96) uint8 tensor → (1,3,96,96) float [0,1] on device."""
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
        if not evaluate:
            action, _, _ = self.policy.sample(img, goal, mot)
        else:
            _, _, action = self.policy.sample(img, goal, mot)
        return action.detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # SAC update
    # ------------------------------------------------------------------

    def update_parameters(self, batch_size):
        imgs, goals, motions, actions, rewards, \
        next_imgs, next_goals, next_motions, dones = self.memory.sample_buffer(batch_size)

        rewards = rewards.unsqueeze(1)
        mask    = (~dones).float().unsqueeze(1)

        with torch.no_grad():
            next_a, next_log_pi, _ = self.policy.sample(next_imgs, next_goals, next_motions)
            q1_next, q2_next = self.critic_target(next_imgs, next_goals, next_motions, next_a)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            next_q = rewards + mask * self.gamma * min_q_next

        q1, q2 = self.critic(imgs, goals, motions, actions)
        critic_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(imgs, goals, motions)
        q1_pi, q2_pi = self.critic(imgs, goals, motions, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

        self.total_grad_steps += 1
        mean_q = torch.min(q1, q2).mean().item()
        return critic_loss.item(), policy_loss.item(), mean_q

    # ------------------------------------------------------------------
    # Run-tag detection (branch-derived, per CLAUDE.md)
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

    def _run_episode(self, collect_only=False, batch_size=64):
        """Run one episode. If collect_only=True, use random actions (warmup)."""
        raw_obs, _ = self.env.reset()
        base = self.env.unwrapped
        r = base._robot
        desired_goal = raw_obs["desired_goal"]
        ms = MotionStateContinuous(self.motion_window)
        obs = self.process_observation(raw_obs["observation"])

        done = False
        episode_reward = 0.0
        episode_steps = 0
        critic_loss_sum = policy_loss_sum = mean_q_sum = 0.0
        update_count = 0

        while not done:
            pos_prev = np.array([r.x, r.y], dtype=np.float32)
            motion_prev = ms.vec(r.x, r.y)
            goal_prev = noisy_world_vector(r.x, r.y, desired_goal[0], desired_goal[1],
                                           self.goal_noise_std)

            if collect_only:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs, goal_prev, motion_prev)

            ms.commit(r.x, r.y, action)
            raw_next, reward, term, trunc, _ = self.env.step(action)
            next_obs = self.process_observation(raw_next["observation"])

            pos_next = np.array([r.x, r.y], dtype=np.float32)
            motion_next = ms.vec(pos_next[0], pos_next[1])
            done = term or trunc
            self.total_env_steps += 1

            self.episode_buffer.store(
                obs, next_obs, action, reward, term,
                achieved_prev=pos_prev, achieved_next=pos_next,
                motion_prev=motion_prev, motion_next=motion_next,
            )
            episode_reward += float(reward)
            episode_steps += 1
            obs = next_obs

            if not collect_only and self.memory.can_sample(batch_size):
                cl, pl, mq = self.update_parameters(batch_size)
                critic_loss_sum += cl
                policy_loss_sum += pl
                mean_q_sum += mq
                update_count += 1

        self.episode_buffer.send_to(
            self.memory, desired_goal=desired_goal,
            compute_reward=base.compute_reward,
            goal_noise_std=self.goal_noise_std,
        )
        self.episode_buffer.clear()
        return episode_reward, episode_steps, critic_loss_sum, policy_loss_sum, mean_q_sum, update_count

    def train(self, episodes=900, batch_size=64, run_tag=None, warmup_steps=5000):
        run_tag = run_tag or self._run_tag()
        writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        warmup_done = 0
        while warmup_done < warmup_steps:
            ep_reward, ep_steps, *_ = self._run_episode(collect_only=True, batch_size=batch_size)
            warmup_done += ep_steps
        if warmup_steps > 0:
            print(f"[warmup] {warmup_done} random steps collected")

        for episode in range(episodes):
            ep_reward, ep_steps, cl_sum, pl_sum, mq_sum, n_updates = \
                self._run_episode(collect_only=False, batch_size=batch_size)

            writer.add_scalar("Train/episode_reward", ep_reward, episode)
            writer.add_scalar("Train/episode_steps",  ep_steps,  episode)
            if n_updates > 0:
                writer.add_scalar("loss/critic",     cl_sum / n_updates, episode)
                writer.add_scalar("loss/policy",     pl_sum / n_updates, episode)
                writer.add_scalar("Train/mean_q",    mq_sum / n_updates, episode)

            print(f"Episode {episode} | reward: {ep_reward:.2f} | steps: {ep_steps}")

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
