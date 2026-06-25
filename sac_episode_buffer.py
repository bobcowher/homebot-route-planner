"""HER relabeling for continuous SAC, adapted from episode_buffer.py.

Differences from episode_buffer.py:
- Goal representation is ego_vector (heading-relative), not noisy_world_vector
  (world-frame) -- matches the continuous [linear, angular] action space.
- The goal vector and motion features are concatenated into ONE flat state
  vector before storage (sac_model's Policy/Critic take a single state input,
  no separate goal-encoder branch), instead of being stored in separate
  goal/motion buffer columns.
- _blocked_penalty does NOT port: "near-zero displacement == wall pin" is
  false for continuous actions, where zero displacement is also the correct
  outcome of a legitimate "stand still" (linear=0) action.
"""
from dataclasses import dataclass
import random

import numpy as np

from goal_geometry import ego_vector


@dataclass
class SACTransition:
    action: np.ndarray
    reward: float
    done: bool
    achieved_prev: np.ndarray
    achieved_next: np.ndarray
    heading_prev: float
    heading_next: float
    motion_prev: np.ndarray
    motion_next: np.ndarray


class SACEpisodeBuffer:
    """Caches one episode's transitions for HER relabeling (future strategy)."""

    K = 2

    def __init__(self):
        self._transitions: list[SACTransition] = []

    def store(self, action, reward, done, achieved_prev, achieved_next,
              heading_prev: float, heading_next: float, motion_prev, motion_next):
        self._transitions.append(SACTransition(
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            done=bool(done),
            achieved_prev=np.asarray(achieved_prev, dtype=np.float32),
            achieved_next=np.asarray(achieved_next, dtype=np.float32),
            heading_prev=float(heading_prev),
            heading_next=float(heading_next),
            motion_prev=np.asarray(motion_prev, dtype=np.float32),
            motion_next=np.asarray(motion_next, dtype=np.float32),
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def _state(self, achieved, heading, goal, motion):
        goal_vec = ego_vector(achieved[0], achieved[1], heading, goal[0], goal[1])
        return np.concatenate([goal_vec, motion]).astype(np.float32)

    def send_to(self, replay_buffer, desired_goal, compute_reward, k: float | None = None) -> None:
        dg = desired_goal
        k = self.K if k is None else k

        for t in self._transitions:
            state = self._state(t.achieved_prev, t.heading_prev, dg, t.motion_prev)
            next_state = self._state(t.achieved_next, t.heading_next, dg, t.motion_next)
            replay_buffer.store_transition(state, t.action, t.reward, next_state, t.done)

        for i, t in enumerate(self._transitions):
            future = self._transitions[i + 1:]
            if not future:
                continue
            kk = int(k) + (1 if random.random() < (k - int(k)) else 0)
            kk = min(kk, len(future))
            if kk <= 0:
                continue
            for hg_t in random.sample(future, kk):
                hindsight_goal = hg_t.achieved_next
                hindsight_reward = float(compute_reward(
                    t.achieved_next[np.newaxis], hindsight_goal[np.newaxis], {},
                )[0])
                hindsight_done = hindsight_reward > 0.5
                hs_state = self._state(t.achieved_prev, t.heading_prev, hindsight_goal, t.motion_prev)
                hs_next_state = self._state(t.achieved_next, t.heading_next, hindsight_goal, t.motion_next)
                replay_buffer.store_transition(hs_state, t.action, hindsight_reward, hs_next_state, hindsight_done)
