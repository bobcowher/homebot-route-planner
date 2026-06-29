"""HER relabeling for discrete SAC, adapted from episode_buffer.py.

Stores raw image tensors per transition (same as the DQN EpisodeBuffer).
Actions are discrete int indices.
HER swaps the goal vector (noisy_world_vector) — images and actions unchanged.

N-STEP RETURNS: each stored transition carries an n-step return
  R = Σ_{k=0}^{m-1} γ^k r_{i+k}   (m = n_step, truncated at a terminal or episode end)
plus the state reached after m steps and a discount γ^m for the bootstrap (the replay
buffer stores that discount per-row; the critic target multiplies V(s_{i+m}) by it).
WHY: under sparse 0/1 reward + γ=0.99 + 1-step bootstrap the per-action advantage
Q(s,a)=r+γV(s') is intrinsically tiny — V has no spatial gradient, so the critic values
all 8 actions near-identically (run 393: entropy pinned at max, Δ≈0). Multi-step returns
propagate the terminal reward up to n steps back, carving a toward-vs-away gradient the
actor can concentrate onto. Standard for sparse goal-conditioned RL; no shaping, no env
change, ports cleanly to the continuous target. Computed here (not in the agent) so HER
relabels get n-step returns too — reward is recomputed per step against each goal.
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

    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self._transitions: list[SACTransition] = []
        self.n_step = int(n_step)
        self.gamma  = float(gamma)

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

    def _nstep_return(self, i: int, goal_xy, compute_reward):
        """Accumulate the n-step return for the transition at index i, evaluating reward
        against goal_xy at every step. Stops early at a terminal (reached goal_xy) or at the
        end of the cached episode. Returns (R, boot, done, discount) where boot is the
        transition whose next_* fields are the bootstrap state s_{i+m}, and discount = γ^m is
        the multiplier on V(boot) (the done mask zeroes the bootstrap on a terminal)."""
        R = 0.0
        disc = 1.0
        done = False
        m = 0
        T = len(self._transitions)
        while m < self.n_step and (i + m) < T:
            t_m = self._transitions[i + m]
            r_m = float(compute_reward(t_m.achieved_next[np.newaxis], goal_xy[np.newaxis], {})[0])
            R += disc * r_m
            disc *= self.gamma
            m += 1
            if r_m > 0.5:          # reached goal_xy -> true terminal, stop bootstrapping
                done = True
                break
        boot = self._transitions[i + m - 1]   # state landed in after m steps; disc == γ^m
        return R, boot, done, disc

    def send_to(self, replay_buffer, desired_goal, compute_reward,
                goal_noise_std: float = 30.0, k: float | None = None) -> None:
        dg = np.asarray(desired_goal, dtype=np.float32)
        k = self.K if k is None else k
        T = len(self._transitions)

        # Real-goal transitions: n-step return toward the true desired goal. Reward is
        # recomputed via compute_reward (the env's own goal-conditioned reward) so the real
        # and hindsight paths share one accumulator.
        for i in range(T):
            t = self._transitions[i]
            R, boot, done, disc = self._nstep_return(i, dg, compute_reward)
            g  = noisy_world_vector(t.achieved_prev[0], t.achieved_prev[1], dg[0], dg[1], goal_noise_std)
            gn = noisy_world_vector(boot.achieved_next[0], boot.achieved_next[1], dg[0], dg[1], goal_noise_std)
            replay_buffer.store_transition(
                t.obs, g, t.motion_prev, t.action, R,
                boot.next_obs, gn, boot.motion_next, done, discount=disc,
            )

        for i in range(T):
            t = self._transitions[i]
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
                hg = np.asarray(future[int(j)].achieved_next, dtype=np.float32)
                R, boot, done, disc = self._nstep_return(i, hg, compute_reward)
                hs_g  = noisy_world_vector(t.achieved_prev[0], t.achieved_prev[1], hg[0], hg[1], goal_noise_std)
                hs_gn = noisy_world_vector(boot.achieved_next[0], boot.achieved_next[1], hg[0], hg[1], goal_noise_std)
                replay_buffer.store_transition(
                    t.obs, hs_g, t.motion_prev, t.action, R,
                    boot.next_obs, hs_gn, boot.motion_next, done, discount=disc,
                )
