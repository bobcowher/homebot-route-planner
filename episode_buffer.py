"""HER relabeling for discrete SAC, adapted from episode_buffer.py.

Stores raw image tensors per transition (same as the DQN EpisodeBuffer).
Actions are discrete int indices.
HER swaps the goal vector (noisy_world_vector) — images and actions unchanged.
"""
from dataclasses import dataclass
import random

import numpy as np
import torch

from goal_geometry import noisy_world_vector


@dataclass
class SACTransition:
    obs:           torch.Tensor   # (3, 96, 96) uint8
    next_obs:      torch.Tensor   # (3, 96, 96) uint8
    action:        int            # discrete action index
    reward:        float
    done:          bool
    achieved_prev: np.ndarray     # robot (x, y) at obs
    achieved_next: np.ndarray     # robot (x, y) at next_obs
    motion_prev:   np.ndarray
    motion_next:   np.ndarray


class SACEpisodeBuffer:
    """Caches one episode's transitions for HER relabeling (future strategy)."""

    K = 2
    HER_HORIZON = 50.0   # future-offset bias scale; ~ gamma's effective horizon (1/(1-gamma))

    def __init__(self):
        self._transitions: list[SACTransition] = []

    def store(self, obs, next_obs, action, reward, done,
              achieved_prev, achieved_next, motion_prev, motion_next):
        self._transitions.append(SACTransition(
            obs=obs,
            next_obs=next_obs,
            action=int(action),
            reward=float(reward),
            done=bool(done),
            achieved_prev=np.asarray(achieved_prev, dtype=np.float32),
            achieved_next=np.asarray(achieved_next, dtype=np.float32),
            motion_prev=np.asarray(motion_prev, dtype=np.float32),
            motion_next=np.asarray(motion_next, dtype=np.float32),
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def send_to(self, replay_buffer, desired_goal, compute_reward,
                goal_noise_std: float = 30.0, k: float | None = None) -> None:
        dg = desired_goal
        k = self.K if k is None else k

        for t in self._transitions:
            g  = noisy_world_vector(t.achieved_prev[0], t.achieved_prev[1], dg[0], dg[1], goal_noise_std)
            gn = noisy_world_vector(t.achieved_next[0], t.achieved_next[1], dg[0], dg[1], goal_noise_std)
            replay_buffer.store_transition(
                t.obs, g, t.motion_prev, t.action, t.reward,
                t.next_obs, gn, t.motion_next, t.done,
            )

        for i, t in enumerate(self._transitions):
            future = self._transitions[i + 1:]
            if not future:
                continue
            kk = int(k) + (1 if random.random() < (k - int(k)) else 0)
            kk = min(kk, len(future))
            if kk <= 0:
                continue
            # Near-biased future sampling: weight each future offset d (1..N) by exp(-d/H) so
            # relabeled goals fall within gamma's effective horizon (~1/(1-gamma)=100 steps at
            # gamma=0.99). Uniform sampling (random.sample) drew mostly FAR goals whose reward
            # is discounted to ~0 -> an advantage the critic can't learn (runs 385/386 stalled
            # with entropy stuck near max). H = HER_HORIZON.
            n_future = len(future)
            offsets = np.arange(1, n_future + 1, dtype=np.float64)
            w = np.exp(-offsets / self.HER_HORIZON)
            w /= w.sum()
            idxs = np.random.choice(n_future, size=kk, replace=False, p=w)
            for j in idxs:
                hg_t = future[int(j)]
                hg = hg_t.achieved_next
                hindsight_reward = float(compute_reward(
                    t.achieved_next[np.newaxis], hg[np.newaxis], {},
                )[0])
                hindsight_done = hindsight_reward > 0.5
                hs_g  = noisy_world_vector(t.achieved_prev[0], t.achieved_prev[1], hg[0], hg[1], goal_noise_std)
                hs_gn = noisy_world_vector(t.achieved_next[0], t.achieved_next[1], hg[0], hg[1], goal_noise_std)
                replay_buffer.store_transition(
                    t.obs, hs_g, t.motion_prev, t.action, hindsight_reward,
                    t.next_obs, hs_gn, t.motion_next, hindsight_done,
                )
