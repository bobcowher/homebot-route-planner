import os
import subprocess
import gymnasium as gym
import torch
import torch.nn.functional as F
import random
import cv2
import datetime
from buffer import ReplayBuffer
from episode_buffer import EpisodeBuffer
from models.q_model import QModel
from torch.utils.tensorboard.writer import SummaryWriter


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 100000,
                       target_update_interval: int = 1000) -> None:
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        raw_obs, _ = self.env.reset()
        obs = self.process_observation(raw_obs["observation"])

        self.n_actions = self.env.action_space.n  # type: ignore[union-attr]

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device=self.device,
            output_device=self.device,
        )

        self.q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
        ).to(self.device)

        self.target_q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
        ).to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.00005)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.977

        self.target_update_interval = target_update_interval
        self.total_steps = 0
        self.episode_buffer = EpisodeBuffer()

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    def select_action(self, obs, goal):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_model(obs_t, goal_t).argmax(dim=1).item()

    def train_step(self, batch_size):
        obs, actions, rewards, next_obs, dones, goals = self.memory.sample_buffer(batch_size)

        obs      = obs      / 255.0
        next_obs = next_obs / 255.0

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones   = dones.unsqueeze(1).float()

        q_sa = self.q_model(obs, goals).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_model(next_obs, goals).argmax(dim=1, keepdim=True)
            next_q       = self.target_q_model(next_obs, goals).gather(1, next_actions)
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

    def save_best(self, score, episode):
        path = "checkpoints/best.pt"
        torch.save({"episode": episode, "score": score, "q_model": self.q_model.state_dict()}, path)
        print(f"Saved best checkpoint | episode: {episode} | score: {score:.1f}")

    def load(self):
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    def train(self, episodes=1000, batch_size=64, run_tag=None):
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

        best_score = float("-inf")

        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]

            done = False
            episode_reward = 0.0
            episode_loss   = 0.0
            episode_steps  = 0

            while not done:
                action = self.select_action(obs, desired_goal)
                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(raw_next["observation"])
                done = term or trunc

                self.episode_buffer.store(
                    obs, action, reward, next_obs, done,
                    achieved_goal=raw_next["achieved_goal"],
                )
                episode_reward += float(reward)
                episode_steps  += 1
                obs = next_obs

            self.episode_buffer.send_to(
                self.memory,
                desired_goal=desired_goal,
                compute_reward=self.env.unwrapped.compute_reward,  # type: ignore[attr-defined]
            )
            self.episode_buffer.clear()

            for _ in range(800):
                if self.memory.can_sample(batch_size):
                    episode_loss += self.train_step(batch_size)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode_reward > best_score:
                best_score = episode_reward
                self.save_best(best_score, episode)

            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/best_score",     best_score,     episode)
            writer.add_scalar("Train/epsilon",         self.epsilon,   episode)
            writer.add_scalar("Train/avg_q_loss",      avg_loss,       episode)
            writer.add_scalar("Train/episode_steps",   episode_steps,  episode)
            writer.add_scalar("Buffer/fill", min(self.memory.mem_ctr, self.memory.mem_size), episode)

            if episode % 10 == 0:
                self.save()

    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []

        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            done = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
                    goal_t = torch.as_tensor(desired_goal, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                    action = self.q_model(obs_t, goal_t).argmax(dim=1).item()
                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(raw_next["observation"])
                done = term or trunc
                episode_reward += float(reward)
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards
