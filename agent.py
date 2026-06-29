"""Discrete SAC + HER agent for HomeBot2D (discrete action mode).

The recipe that learns full random-start navigation:
  - Double-Q critic + categorical actor (Christodoulou 2019 discrete SAC update).
  - HER on the episode buffer (relabel to achieved goals) — HER *is* the curriculum.
  - Fixed temperature alpha (no auto-tuning — it never converged usefully here).
  - Behaviour = SAMPLE a ~ π(·|s) from the stochastic actor. A sampled soft policy cannot
    lock into the deterministic A<->B oscillation that argmax-over-critic (= DQN) falls
    into; the actor's own entropy is the exploration (no epsilon-greedy). This is the whole
    reason for actor-critic here. The critic only has to give the actor a usable gradient —
    it does NOT need to be argmax-reachable.

Observation pipeline:
  image  : 96x96 RGB -> permute(2,0,1) -> uint8
  goal   : noisy_world_vector(rx, ry, gx, gy, noise_std) -> [dx, dy] float32
  motion : MotionStateDiscrete -> [dx/step, dy/step, 0, 0] float32

Discrete SAC update (HARD-VALUE bootstrap variant — entropy in the actor, NOT the target):
  V(s')  = Σ_a π(a|s')·min_Q_target(s', a)           (NO −α logπ — see update_parameters)
  critic : MSE(Q(s, a), R_n + γ^m(1-done)·V(s_{t+m}))  (plain MSE, polyak target tau=0.005)
  actor  : Σ_a π(a|s)[α logπ(a|s) - min_Q(s, a)]     (entropy kept here -> policy stays soft)
The canonical soft target's α·H/(1−γ) entropy offset floods mean_q to ~10 and buries the
HER goal-advantage (run 389 image-blind diag: zeroing the image didn't help -> flood is in
the bootstrap, not the representation). Dropping it from the target keeps Q at true-return
scale while the actor stays stochastic.

N-STEP RETURNS: R_n and the m-step bootstrap discount γ^m are precomputed per transition in
the episode buffer (so HER relabels get them too) and stored with each replay row; the target
above just reads them. Multi-step returns propagate the sparse terminal reward up to n steps
back, carving the toward-vs-away value gradient that 1-step bootstrapping left flat (run 393:
Δ≈0, actor entropy pinned at max). See episode_buffer.SACEpisodeBuffer._nstep_return.
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
                 goal_noise_std=30.0, head_layers=4, head_hidden=512, n_step=3):
        self.env = env
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha            # fixed entropy temperature (canonical SAC; static value)
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

        self.n_step = int(n_step)     # n-step return horizon (see episode_buffer)
        self.memory = SACReplayBuffer(max_buffer_size, device=str(self.device))
        self.episode_buffer = SACEpisodeBuffer(n_step=self.n_step, gamma=gamma)
        self.total_env_steps = 0
        self.total_grad_steps = 0
        self._explore_heading = None   # persistent heading for front-biased warmup walk

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
        """argmax_a min(q1, q2)(s, a) — the OLD DQN-in-costume behaviour. Kept for eval
        comparison only; the live policy now samples the actor (sample_actor_action)."""
        img  = self._to_device_float(obs_tensor)
        goal = self._goal_tensor(goal_np)
        mot  = self._motion_tensor(motion_np)
        with torch.no_grad():
            q1, q2 = self.critic(img, goal, mot)
            return int(torch.min(q1, q2).argmax(dim=-1).item())

    def sample_actor_action(self, obs_tensor, goal_np, motion_np, greedy=False):
        """Behaviour: sample a ~ π(·|s) from the stochastic actor (greedy=True -> argmax π
        for deterministic eval). Sampling is what breaks the deterministic A<->B oscillation
        that drove us off DQN, and the actor's entropy supplies exploration."""
        img  = self._to_device_float(obs_tensor)
        goal = self._goal_tensor(goal_np)
        mot  = self._motion_tensor(motion_np)
        with torch.no_grad():
            probs, _ = self.policy(img, goal, mot)
            if greedy:
                return int(probs.argmax(dim=-1).item())
            return int(torch.distributions.Categorical(probs=probs).sample().item())

    def _biased_explore_action(self):
        """Front-biased warmup explorer. The 8 discrete actions are ABSOLUTE compass dirs
        ordered around a ring (homebot.robot._DIRS: 0=N,1=NE,...,7=NW, 45° apart), and they
        are symmetric — uniform-random has ZERO mean displacement (N/S, E/W cancel), so the
        bot wobbles in place and HER gets a degenerate near-stationary trajectory. Instead
        keep a persistent heading and draw the next move from the forward 120° arc around it:
        50% straight ahead, 50% split across gentle (±1) / hard (±2) turns. NEVER the reverse
        (h+4) or backward-leaning (h±3) dirs. Yields directed, map-covering walks that give
        HER real trajectory to relabel. (Discrete analog of the diff-drive 'forward + turns,
        never back' prior — maps literally onto the continuous action space later. Relies on
        the action indices being ring-ordered, which _DIRS is.)
        """
        n = self.n_actions
        if self._explore_heading is None:
            self._explore_heading = random.randrange(n)
        offsets = (0, 1, -1, 2, -2)            # forward, gentle turns, hard turns
        weights = (0.50, 0.15, 0.15, 0.10, 0.10)
        off = random.choices(offsets, weights=weights)[0]
        self._explore_heading = (self._explore_heading + off) % n
        return self._explore_heading

    # ------------------------------------------------------------------
    # Discrete SAC update
    # ------------------------------------------------------------------

    def update_parameters(self, batch_size):
        imgs, goals, motions, actions, rewards, \
        next_imgs, next_goals, next_motions, dones, discounts = self.memory.sample_buffer(batch_size)

        rewards   = rewards.unsqueeze(1)
        mask      = (~dones).float().unsqueeze(1)
        discounts = discounts.unsqueeze(1)   # γ^m bootstrap multiplier (n-step; see episode_buffer)

        # Critic target: HARD value V(s') = Σ_a π(a|s')·min_Q_target(s', a) — the entropy
        # term (−α·logπ) is DELIBERATELY DROPPED from the bootstrap (vs canonical soft SAC).
        # WHY (run 389, image-blind diagnostic): with the soft bootstrap the entropy offset
        # α·Σπ·(−logπ) ≈ α·H/(1−γ) floods mean_q to ~10 (true max ~1) — a ~uniform ~10-magnitude
        # term that BURIES the small inter-action goal-advantage HER manufactures, so the actor
        # never gets a usable gradient and stays uniform. Zeroing the image (389) didn't help ->
        # the flood is in the BOOTSTRAP, not the representation. Removing −α·logπ here pulls the
        # target back to true-return scale (~1) so the advantage survives. Entropy is NOT
        # abandoned — it stays in the ACTOR loss below, so the policy is still stochastic
        # (exploration + anti-A<->B-oscillation, the reason we run SAC at all). This decouples
        # "keep the policy soft" (actor objective) from "don't flood Q" (critic target).
        with torch.no_grad():
            next_probs, _ = self.policy(next_imgs, next_goals, next_motions)
            q1_next, q2_next = self.critic_target(next_imgs, next_goals, next_motions)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * min_q_next).sum(dim=1, keepdim=True)
            # rewards is the n-step return Σγ^k r; discounts is γ^m for the m-step bootstrap
            # (m truncated at a terminal/episode end), so γ is NOT applied again here.
            target_q = rewards + mask * discounts * v_next

        # Critic loss: plain MSE of each head against the soft target (canonical, no Q-clip).
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

        # Polyak (soft) target update every grad-step — canonical SAC, matching the ant-maze
        # reference (tau=0.005). (Reverted from hard-sync@1000; the earlier polyak divergence
        # was under argmax-over-critic behaviour + the other patches, not the canonical core.)
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

    def _run_episode(self, collect_only=False, batch_size=64, directed_episode=False):
        raw_obs, _ = self.env.reset()
        base = self.env.unwrapped
        r = base._robot
        desired_goal = raw_obs["desired_goal"]
        ms  = MotionStateDiscrete()
        self._explore_heading = None   # fresh random bearing per warmup episode
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

            # Behaviour: an entire episode is either a directed-walk traversal or actor-driven.
            # Per-EPISODE (not per-step) is the fix for run 382: a per-step mix let the actor's
            # wobble dominate the trajectory so it never crossed the map. A whole directed
            # episode gives HER a clean map-crossing, goal-reaching trajectory to relabel —
            # the real far-goal advantage the actor needs. Warmup is always directed.
            if collect_only or directed_episode:
                action = self._biased_explore_action()
            else:
                action = self.sample_actor_action(obs, goal_prev, motion_prev)

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
              explore_start=1.0, explore_min=0.25, explore_decay=0.977):
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
            # Decaying fraction of WHOLE episodes are pure directed traversals (Q-learning
            # schedule): heavy early to seed the critic with map-crossing, goal-reaching
            # trajectories for HER, fading to a floor (~0.25) so ~1/4 of episodes keep feeding
            # real far-goal data. The remaining episodes are pure actor — that's where we read
            # the policy's true reach.
            explore_rate = max(explore_min, explore_start * (explore_decay ** episode))
            directed_episode = random.random() < explore_rate
            ep_reward, ep_steps, cl_sum, al_sum, mq_sum, ent_sum, n_updates = \
                self._run_episode(collect_only=False, batch_size=batch_size,
                                  directed_episode=directed_episode)

            writer.add_scalar("Train/episode_reward", ep_reward, episode)
            writer.add_scalar("Train/episode_steps",  ep_steps,  episode)
            writer.add_scalar("Train/explore_rate",   explore_rate, episode)
            writer.add_scalar("Train/directed_episode", float(directed_episode), episode)
            if n_updates > 0:
                writer.add_scalar("loss/critic",          cl_sum  / n_updates, episode)
                writer.add_scalar("loss/actor",           al_sum  / n_updates, episode)
                writer.add_scalar("Train/mean_q",         mq_sum  / n_updates, episode)
                writer.add_scalar("Train/policy_entropy", ent_sum / n_updates, episode)

            tag = "DIR" if directed_episode else "act"
            print(f"Episode {episode} [{tag}] | reward: {ep_reward:.2f} | steps: {ep_steps} | "
                  f"explore: {explore_rate:.3f} | entropy: {ent_sum / max(n_updates, 1):.3f}")
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
