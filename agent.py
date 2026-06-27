"""Discrete SAC + HER agent for HomeBot2D (discrete action mode).

The recipe that learns full random-start navigation:
  - Double-Q critic + categorical actor (Christodoulou 2019 discrete SAC update).
  - HER on the episode buffer (relabel to achieved goals) — HER *is* the curriculum.
  - Fixed temperature alpha (no auto-tuning — it never converged usefully here).
  - Behaviour = epsilon-greedy ARGMAX over the CRITIC (min Q), not the actor. The HER
    critic holds the small inter-action goal-advantage that the entropy-regularised actor
    washes out; argmax follows it regardless of magnitude (the DQN mechanism). The actor
    only feeds the soft-value bootstrap.

Observation pipeline:
  image  : 96x96 RGB -> permute(2,0,1) -> uint8
  goal   : noisy_world_vector(rx, ry, gx, gy, noise_std) -> [dx, dy] float32
  motion : MotionStateDiscrete -> [dx/step, dy/step, 0, 0] float32

Discrete SAC update:
  V(s')  = Σ_a π(a|s')[Q_target(s', a) - α logπ(a|s')]
  critic : MSE(Q(s, a), r + γ(1 - done)·V(s'))
  actor  : Σ_a π(a|s)[α logπ(a|s) - min_Q(s, a)]
"""
import cv2
import datetime
import os
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from goal_geometry import noisy_world_vector
from buffer import SACReplayBuffer
from episode_buffer import SACEpisodeBuffer
from model import DiscreteQNet, DiscretePolicy
from motion import MotionStateDiscrete


class SACAgent:
    def __init__(self, env, max_buffer_size=200000,
                 gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4,
                 goal_noise_std=30.0, head_layers=4, head_hidden=512):
        self.env = env
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha            # fixed entropy temperature (a mild actor regulariser)
        self.goal_noise_std = goal_noise_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        self.critic = DiscreteQNet(self.n_actions, name="critic",
                                   head_layers=head_layers, head_hidden=head_hidden).to(self.device)
        self.critic_target = DiscreteQNet(self.n_actions, name="critic_target",
                                          head_layers=head_layers, head_hidden=head_hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.policy = DiscretePolicy(self.n_actions, name="actor",
                                     head_layers=head_layers, head_hidden=head_hidden).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.memory = SACReplayBuffer(max_buffer_size, device=str(self.device))
        self.episode_buffer = SACEpisodeBuffer()
        self.total_env_steps = 0
        self.total_grad_steps = 0

    # ------------------------------------------------------------------
    # Observation pipeline + tensor helpers
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

    def greedy_critic_action(self, obs_tensor, goal_np, motion_np):
        """argmax_a min(q1, q2)(s, a) — behaviour comes from the CRITIC, not the actor."""
        img  = self._to_device_float(obs_tensor)
        goal = self._goal_tensor(goal_np)
        mot  = self._motion_tensor(motion_np)
        with torch.no_grad():
            q1, q2 = self.critic(img, goal, mot)
            return int(torch.min(q1, q2).argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # Discrete SAC update
    # ------------------------------------------------------------------

    def update_parameters(self, batch_size):
        imgs, goals, motions, actions, rewards, \
        next_imgs, next_goals, next_motions, dones = self.memory.sample_buffer(batch_size)

        rewards = rewards.unsqueeze(1)
        mask    = (~dones).float().unsqueeze(1)

        # Critic target: soft value V(s') over the discrete action expectation.
        with torch.no_grad():
            next_probs, next_log_probs = self.policy(next_imgs, next_goals, next_motions)
            q1_next, q2_next = self.critic_target(next_imgs, next_goals, next_motions)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + mask * self.gamma * v_next

        # Critic loss: Q for the taken action vs target.
        q1, q2 = self.critic(imgs, goals, motions)
        q1_a = q1.gather(1, actions.unsqueeze(1))
        q2_a = q2.gather(1, actions.unsqueeze(1))
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss: exact expectation over actions, L = Σ_a π[α logπ - min_Q].
        probs, log_probs = self.policy(imgs, goals, motions)
        with torch.no_grad():
            q1, q2 = self.critic(imgs, goals, motions)
        min_q = torch.min(q1, q2)
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # Polyak target update.
        for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

        self.total_grad_steps += 1
        mean_q  = min_q.mean().item()
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean().item()
        return critic_loss.item(), actor_loss.item(), mean_q, entropy

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

    def _run_episode(self, collect_only=False, batch_size=64, epsilon=0.0):
        raw_obs, _ = self.env.reset()
        base = self.env.unwrapped
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

            # Epsilon-greedy argmax over the critic.
            if collect_only or random.random() < epsilon:
                action = int(self.env.action_space.sample())
            else:
                action = self.greedy_critic_action(obs, goal_prev, motion_prev)

            ms.commit(r.x, r.y, action)
            raw_next, reward, term, trunc, _ = self.env.step(action)
            next_obs = self.process_observation(raw_next["observation"])
            pos_next    = np.array([r.x, r.y], dtype=np.float32)
            motion_next = ms.vec(pos_next[0], pos_next[1])
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

        # HER: relabel each transition's goal to achieved future positions. This is the
        # curriculum — the agent is always trained on goals it actually reached.
        self.episode_buffer.send_to(
            self.memory, desired_goal=desired_goal,
            compute_reward=base.compute_reward,
            goal_noise_std=self.goal_noise_std,
        )
        self.episode_buffer.clear()
        return (episode_reward, episode_steps,
                critic_loss_sum, actor_loss_sum, mean_q_sum, entropy_sum, update_count)

    def train(self, episodes=1200, batch_size=64, run_tag=None, warmup_steps=5000,
              epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.977):
        run_tag = run_tag or self._run_tag()
        writer  = SummaryWriter(
            f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        # Warmup: fill the buffer with random transitions before any gradient update.
        warmup_done = 0
        while warmup_done < warmup_steps:
            _, ep_steps, *_ = self._run_episode(collect_only=True, batch_size=batch_size)
            warmup_done += ep_steps
        if warmup_steps > 0:
            print(f"[warmup] {warmup_done} random steps collected")

        for episode in range(episodes):
            epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
            ep_reward, ep_steps, cl_sum, al_sum, mq_sum, ent_sum, n_updates = \
                self._run_episode(collect_only=False, batch_size=batch_size, epsilon=epsilon)

            writer.add_scalar("Train/episode_reward", ep_reward, episode)
            writer.add_scalar("Train/episode_steps",  ep_steps,  episode)
            writer.add_scalar("Train/epsilon",        epsilon,   episode)
            if n_updates > 0:
                writer.add_scalar("loss/critic",          cl_sum  / n_updates, episode)
                writer.add_scalar("loss/actor",           al_sum  / n_updates, episode)
                writer.add_scalar("Train/mean_q",         mq_sum  / n_updates, episode)
                writer.add_scalar("Train/policy_entropy", ent_sum / n_updates, episode)

            print(f"Episode {episode} | reward: {ep_reward:.2f} | "
                  f"steps: {ep_steps} | eps: {epsilon:.3f}")
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
