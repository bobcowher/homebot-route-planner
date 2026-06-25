"""Continuous SAC agent for HomeBotGoalEnv. Engine (select_action,
update_parameters) ported from sac-fetch/agent.py; train() is a new HER
training loop wiring sac_episode_buffer + sac_motion onto HomeBotGoalEnv's
continuous action_mode."""
import datetime
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from goal_geometry import ego_vector
from sac_buffer import SACReplayBuffer
from sac_episode_buffer import SACEpisodeBuffer
from sac_model import Critic, Policy
from sac_motion import MotionStateContinuous


class SACAgent:
    def __init__(self, env, state_dim, action_dim, max_buffer_size=200000,
                 hidden_dim=128, gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4,
                 motion_window=1):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.motion_window = motion_window
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        self.critic = Critic(state_dim, action_dim, hidden_dim, name="sac_critic").to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, name="sac_critic_target").to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.policy = Policy(state_dim, action_dim, hidden_dim, env.action_space, name="sac_policy").to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.memory = SACReplayBuffer(max_buffer_size, state_dim, action_dim)
        self.episode_buffer = SACEpisodeBuffer()

        self.total_env_steps = 0
        self.total_grad_steps = 0

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.policy.sample(state_t)
        else:
            _, _, action = self.policy.sample(state_t)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):
        state, action, reward, next_state, done = self.memory.sample_buffer(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        mask = torch.FloatTensor(1.0 - done.astype(np.float32)).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            next_q = reward + mask * self.gamma * min_q_next

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_grad_steps += 1
        mean_q = torch.min(q1, q2).mean().item()
        return critic_loss.item(), policy_loss.item(), mean_q

    def _build_state(self, rx, ry, rtheta, gx, gy, motion):
        goal_vec = ego_vector(rx, ry, rtheta, gx, gy)
        return np.concatenate([goal_vec, motion]).astype(np.float32)

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

    def train(self, episodes=1800, batch_size=64, run_tag=None, warmup_steps=5000):
        run_tag = run_tag or self._run_tag()
        writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        warmup_done = 0
        while warmup_done < warmup_steps:
            raw_obs, _ = self.env.reset()
            base = self.env.unwrapped
            r = base._robot
            desired_goal = raw_obs["desired_goal"]
            ms = MotionStateContinuous(self.motion_window)
            done = False
            while not done:
                heading_prev = r.angle
                pos_prev = np.array([r.x, r.y], dtype=np.float32)
                motion_prev = ms.vec(r.x, r.y)
                action = self.env.action_space.sample()
                ms.commit(r.x, r.y, action)
                _, reward, term, trunc, _ = self.env.step(action)
                pos_next = np.array([r.x, r.y], dtype=np.float32)
                heading_next = r.angle
                motion_next = ms.vec(pos_next[0], pos_next[1])
                done = term or trunc
                self.total_env_steps += 1
                warmup_done += 1
                self.episode_buffer.store(
                    action, reward, term,
                    achieved_prev=pos_prev, achieved_next=pos_next,
                    heading_prev=heading_prev, heading_next=heading_next,
                    motion_prev=motion_prev, motion_next=motion_next,
                )
            self.episode_buffer.send_to(
                self.memory, desired_goal=desired_goal, compute_reward=base.compute_reward,
            )
            self.episode_buffer.clear()
        if warmup_steps > 0:
            print(f"[warmup] {warmup_done} random steps collected")

        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            base = self.env.unwrapped
            r = base._robot
            desired_goal = raw_obs["desired_goal"]
            ms = MotionStateContinuous(self.motion_window)

            done = False
            episode_reward = 0.0
            episode_steps = 0
            critic_loss_sum = policy_loss_sum = mean_q_sum = 0.0
            update_count = 0

            while not done:
                heading_prev = r.angle
                pos_prev = np.array([r.x, r.y], dtype=np.float32)
                motion_prev = ms.vec(r.x, r.y)
                state = self._build_state(r.x, r.y, r.angle,
                                          desired_goal[0], desired_goal[1], motion_prev)

                action = self.select_action(state)
                ms.commit(r.x, r.y, action)
                _, reward, term, trunc, _ = self.env.step(action)

                pos_next = np.array([r.x, r.y], dtype=np.float32)
                heading_next = r.angle
                motion_next = ms.vec(pos_next[0], pos_next[1])
                done = term or trunc
                self.total_env_steps += 1

                # Store term (not trunc): a timeout isn't a terminal state, so the
                # target should still bootstrap from next_state.
                self.episode_buffer.store(
                    action, reward, term,
                    achieved_prev=pos_prev, achieved_next=pos_next,
                    heading_prev=heading_prev, heading_next=heading_next,
                    motion_prev=motion_prev, motion_next=motion_next,
                )
                episode_reward += float(reward)
                episode_steps += 1

                if self.memory.can_sample(batch_size):
                    critic_loss, policy_loss, mean_q = self.update_parameters(batch_size)
                    critic_loss_sum += critic_loss
                    policy_loss_sum += policy_loss
                    mean_q_sum += mean_q
                    update_count += 1

            self.episode_buffer.send_to(
                self.memory, desired_goal=desired_goal, compute_reward=base.compute_reward,
            )
            self.episode_buffer.clear()

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/episode_steps", episode_steps, episode)
            if update_count > 0:
                writer.add_scalar("loss/critic", critic_loss_sum / update_count, episode)
                writer.add_scalar("loss/policy", policy_loss_sum / update_count, episode)
                # The literal failure signature from goal_reacher_overestimation.md --
                # watch this for runaway growth, not the reward curve.
                writer.add_scalar("Train/mean_q", mean_q_sum / update_count, episode)

            print(f"Episode {episode} | reward: {episode_reward:.2f} | steps: {episode_steps}")

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
