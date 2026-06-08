import os
import subprocess
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
from buffer import ReplayBuffer
from utils import hard_update
from models.actor import Actor
from models.critic import Critic
from torch.optim import Adam
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime


class Agent:

    def __init__(self, env : gym.Env,
                       max_buffer_size : int = 10000,
                       target_update_interval = 10000,
                       alpha : float = 0.1,
                       tau : float = 0.005,
                       n_step: int = 5) -> None:
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.critic_lr = 0.0001
        self.actor_lr = 3e-5
        self.alpha = alpha
        self.n_step = n_step

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        obs, _ = self.env.reset()
        obs = self.process_observation(obs)

        # Actor operates in a structured 2D action space [steering, throttle_brake].
        # throttle_brake ∈ [-1, 1]: positive → gas, negative → brake.
        # decode_action() maps this to the 3D env action before env.step().
        self.actor_action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self.n_actions = self.actor_action_space.shape[0]  # 2
        self.ac_hidden_size = 256

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device='cpu',
            output_device=self.device,
            action_dim=self.n_actions,
        )

        print(f"Observation shape: {obs.shape}")

        # obs is (C, H, W); three stride-2 convs → (128, H//8, W//8)
        _, H, W = obs.shape
        conv_flat_size = 128 * (H // 8) * (W // 8)

        self.critic = Critic(num_inputs=conv_flat_size,
                             num_actions=self.n_actions,
                             hidden_dim=self.ac_hidden_size,
                             name="critic").to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.critic_target = Critic(num_inputs=conv_flat_size,
                                    num_actions=self.n_actions,
                                    hidden_dim=self.ac_hidden_size,
                                    name="critic_target").to(self.device)

        hard_update(self.critic_target, self.critic)

        self.actor = Actor(num_inputs=conv_flat_size,
                           num_actions=self.n_actions,
                           hidden_dim=self.ac_hidden_size,
                           action_space=self.actor_action_space,
                           name="policy").to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)

        self.target_update_interval = target_update_interval

        self.gamma = 0.99
        self.tau = tau

        self.total_steps = 0

    def decode_action(self, actor_action: np.ndarray) -> np.ndarray:
        steering = float(actor_action[0])
        tb       = float(actor_action[1])
        gas      = max(0.0, tb)
        brake    = max(0.0, -tb)
        return np.array([steering, gas, brake], dtype=np.float32)

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    def train_actor_critic(self, batch_size):
        if not self.memory.can_sample(batch_size):
            return None

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            self.memory.sample_nstep(batch_size, self.n_step, self.gamma)

        state_batch      = (state_batch.float()      / 255.0).to(self.device)
        next_state_batch = (next_state_batch.float() / 255.0).to(self.device)
        action_batch     = action_batch.float().to(self.device)
        reward_batch     = reward_batch.float().to(self.device).unsqueeze(1)
        mask_batch       = mask_batch.float().to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * (self.gamma ** self.n_step) * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item()

    def save(self):
        self.actor.save_the_model("actor", verbose=True)
        self.critic.save_the_model("critic", verbose=True)

    def save_best(self, score, episode):
        path = "checkpoints/best.pt"
        torch.save({
            "episode": episode,
            "score": score,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Saved best checkpoint to {path} | episode: {episode} | score: {score:.1f}")

    def load(self):
        self.actor.load_the_model("actor", device=self.device)
        self.critic.load_the_model("critic", device=self.device)
        hard_update(self.critic_target, self.critic)

    def test(self, episodes=10):
        self.actor.eval()
        total_rewards = []

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)
            done = False
            episode_reward = 0.0

            while not done:
                obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
                actor_action = self.select_action(obs_t, evaluate=True)
                next_obs, reward, term, trunc, _ = self.env.step(self.decode_action(actor_action))
                next_obs = self.process_observation(next_obs)
                done = term or trunc
                episode_reward += float(reward)
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.actor.train()
        return total_rewards

    def select_action(self, state, evaluate=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.float().to(self.device)
        if state.dim() < 4:
            state = state.unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def warmup_action(self) -> np.ndarray:
        steering       = np.random.uniform(-1.0, 1.0)
        throttle_brake = np.random.uniform(0.0, 1.0)
        return np.array([steering, throttle_brake], dtype=np.float32)

    def train(self, episodes=1, batch_size=32, warmup_episodes=5, run_tag=None):

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

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}'
        writer = SummaryWriter(summary_writer_name)

        best_score = float("-inf")

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)

            done = False
            episode_reward = 0.0
            episode_steps = 0
            total_qf1_loss  = 0.0
            total_qf2_loss  = 0.0
            total_actor_loss = 0.0
            ac_updates = 0

            while not done:
                if episode < warmup_episodes:
                    actor_action = self.warmup_action()
                else:
                    obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
                    actor_action = self.select_action(obs_t)

                car_action = self.decode_action(actor_action)
                next_obs, reward, term, trunc, _ = self.env.step(car_action)

                next_obs     = self.process_observation(next_obs)
                done         = term or trunc
                episode_done = term or trunc

                self.memory.store_transition(obs, actor_action, reward, next_obs, term, episode_done)

                episode_reward += float(reward)
                episode_steps  += 1
                obs = next_obs

                # One AC update per env step after warmup
                if episode >= warmup_episodes:
                    result = self.train_actor_critic(batch_size)
                    if result is not None:
                        qf1_loss, qf2_loss, actor_loss = result
                        total_qf1_loss  += qf1_loss
                        total_qf2_loss  += qf2_loss
                        total_actor_loss += actor_loss
                        ac_updates += 1

            print(f"Episode {episode} | reward: {episode_reward:.1f} | steps: {episode_steps}")

            if episode_reward > best_score:
                best_score = episode_reward
                self.save_best(best_score, episode)

            if episode % 10 == 0:
                hard_update(self.critic_target, self.critic)

            episode_loss = (total_qf1_loss + total_qf2_loss) / (2 * ac_updates) if ac_updates > 0 else 0.0

            if ac_updates > 0:
                writer.add_scalar("SAC/qf1_loss",   total_qf1_loss   / ac_updates, episode)
                writer.add_scalar("SAC/qf2_loss",   total_qf2_loss   / ac_updates, episode)
                writer.add_scalar("SAC/actor_loss", total_actor_loss / ac_updates, episode)

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/avg_critic_loss", episode_loss, episode)
            writer.add_scalar("Train/best_score", best_score, episode)
            writer.add_scalar("Train/episode_steps", episode_steps, episode)
            writer.add_scalar("Buffer/fill", min(self.memory.mem_ctr, self.memory.mem_size), episode)

            if episode % 10 == 0:
                self.save()
