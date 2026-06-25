# Continuous SAC + HER Re-baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a continuous-action SAC+HER agent on `HomeBotGoalEnv` (`action_mode="continuous"`) that trains stably (no value-overestimation collapse) on the `collect_trash` leg.

**Architecture:** Port `sac-fetch`'s vanilla double-Q SAC (`Policy`/`Critic`, MLP, fixed alpha) unmodified into new `sac_*.py` files in this repo. Layer HER (adapted from this repo's `episode_buffer.py`) on top, using `goal_geometry.ego_vector` (heading-relative goal vector, already exists) as the sole observation feature — no image/CNN, no n-step, no macro-actions for v1.

**Tech Stack:** PyTorch, Gymnasium, `gym-homebot-2d` (`HomeBot2D-Goal-V1`), TensorBoard.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-24-sac-continuous-her-design.md`
- Success bar is **stability, not parity**: no overestimation collapse. Matching champion-314/run325 numbers is explicitly out of scope for this plan.
- No CNN/image observation. No n-step returns. No macro-actions. No goal noise (`noisy_ego_vector`). All deferred per spec.
- v1 trains on `goals=["collect_trash"], n_trash=1, random_start=True` only — mirrors current `train.py` exactly.
- `_blocked_penalty` (from `episode_buffer.py`) does NOT port — its "zero displacement == wall pin" assumption is false for continuous actions (zero `linear` is a legitimate no-op).
- Do not modify `motion.py`, `episode_buffer.py`, `buffer.py`, `agent.py`, `models/q_model.py`, or any other file the discrete champion depends on. All new code lives in new `sac_*.py` files so the existing discrete training path keeps working unmodified on this branch.
- Fixed alpha (no auto-entropy tuning) — matches `sac_fixed_alpha.md`.

---

### Task 1: Fix the stale `gym-homebot-2d` pin

**Files:**
- Modify: `requirements.txt:13`

**Interfaces:**
- Produces: working `homebot.env.HomeBotGoalEnv` import with the real-TaskManager-reward fix (`bf0dba6`) available to every later task.

- [ ] **Step 1: Confirm the current pin is stale**

Run: `pip show -f gym-homebot-2d | grep -A1 direct_url` — or simpler, check the installed version's env.py for the bug:

```bash
python3 -c "import homebot.env, inspect; print('HomeBotGoalEnv' in dir(homebot.env))"
```

This is allowed as a one-off diagnostic (not an implementation step), but since inline `-c` execution prompts every time, write it to a scratch file instead and run that:

```python
# /tmp/claude-1000/-home-robertcowher-pythonprojects-sac-homebot/97f0b378-d6d1-4f5e-85f8-8fb4107c1551/scratchpad/check_goal_env.py
import homebot.env
print("HomeBotGoalEnv present:", hasattr(homebot.env, "HomeBotGoalEnv"))
```

Run: `python3 /tmp/claude-1000/-home-robertcowher-pythonprojects-sac-homebot/97f0b378-d6d1-4f5e-85f8-8fb4107c1551/scratchpad/check_goal_env.py`
Expected: `HomeBotGoalEnv present: False` (the installed snapshot predates it) — confirms the reinstall is needed.

- [ ] **Step 2: Bump the pin**

In `requirements.txt`, change line 13 from:
```
git+https://github.com/bobcowher/gym-homebot-2d.git@de9c437bf12980b58bf287c47d2085b5f4c21760
```
to:
```
git+https://github.com/bobcowher/gym-homebot-2d.git@bf0dba69834841a6dd7ebb2bd26671b84cd6f538
```

- [ ] **Step 3: Reinstall and verify**

Run: `pip install --upgrade --force-reinstall --no-deps "git+https://github.com/bobcowher/gym-homebot-2d.git@bf0dba69834841a6dd7ebb2bd26671b84cd6f538"`

Then re-run the same scratch script from Step 1.
Expected: `HomeBotGoalEnv present: True`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "fix: pin gym-homebot-2d to bf0dba6 (real-TaskManager-reward fix)"
```

---

### Task 2: `sac_motion.py` — continuous-action motion features

**Files:**
- Create: `sac_motion.py`
- Test: `tests/test_sac_motion.py`

**Interfaces:**
- Consumes: `goal_geometry.ROBOT_STEP_PX` (existing constant, value `4.0`)
- Produces: `motion_dim_continuous(window: int = 1) -> int`, `make_motion_continuous(last_action, dx, dy, net_dx=0.0, net_dy=0.0, step=ROBOT_STEP_PX, window=1) -> np.ndarray`, `class MotionStateContinuous: __init__(self, window: int = 1)`, `.reset()`, `.vec(x, y) -> np.ndarray`, `.commit(x, y, action)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sac_motion.py
import numpy as np
from sac_motion import motion_dim_continuous, make_motion_continuous, MotionStateContinuous


def test_motion_dim_window_1():
    assert motion_dim_continuous(window=1) == 4  # [last_lin, last_ang, dx, dy]


def test_motion_dim_window_gt_1():
    assert motion_dim_continuous(window=8) == 6  # + net_dx, net_dy


def test_make_motion_continuous_first_step_no_last_action():
    m = make_motion_continuous(last_action=None, dx=4.0, dy=0.0, step=4.0, window=1)
    assert m.shape == (4,)
    assert np.allclose(m, [0.0, 0.0, 1.0, 0.0])


def test_make_motion_continuous_with_last_action():
    m = make_motion_continuous(last_action=np.array([0.5, -1.0]), dx=2.0, dy=2.0, step=4.0, window=1)
    assert np.allclose(m, [0.5, -1.0, 0.5, 0.5])


def test_make_motion_continuous_windowed():
    m = make_motion_continuous(last_action=np.array([1.0, 0.0]), dx=4.0, dy=0.0,
                                net_dx=8.0, net_dy=0.0, step=4.0, window=8)
    assert m.shape == (6,)
    assert np.allclose(m, [1.0, 0.0, 1.0, 0.0, 0.25, 0.0])


def test_motion_state_continuous_first_vec_is_zero_motion():
    ms = MotionStateContinuous(window=1)
    m = ms.vec(100.0, 100.0)
    assert np.allclose(m, [0.0, 0.0, 0.0, 0.0])


def test_motion_state_continuous_tracks_displacement_and_last_action():
    ms = MotionStateContinuous(window=1)
    ms.vec(100.0, 100.0)
    ms.commit(100.0, 100.0, np.array([1.0, 0.0]))
    m = ms.vec(104.0, 100.0)
    assert np.allclose(m, [1.0, 0.0, 1.0, 0.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sac_motion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sac_motion'`

- [ ] **Step 3: Write the implementation**

```python
# sac_motion.py
"""Continuous-action analog of motion.py's anti-oscillation features.

motion.py's last-action term is a one-hot sized by the discrete action_dim,
which doesn't apply to a continuous [linear, angular] action. This module
keeps the same windowed-net-displacement idea (still a valid spin signal for
a heading-controlled policy) but represents the previous action as its raw
2-vector instead of a one-hot index.
"""
from collections import deque

import numpy as np

from goal_geometry import ROBOT_STEP_PX


def motion_dim_continuous(window: int = 1) -> int:
    """[last_linear, last_angular, dx, dy] plus [net_dx, net_dy] when window > 1."""
    return 2 + 2 + (2 if window > 1 else 0)


def make_motion_continuous(last_action, dx, dy, net_dx=0.0, net_dy=0.0,
                            step=ROBOT_STEP_PX, window=1):
    """[last_action(2) | dx/step | dy/step | net_dx/(W*step) | net_dy/(W*step)].

    last_action None -> zeros (episode start). Velocity normalized by the max
    per-step speed so it sits in ~[-1, 1], same convention as motion.py."""
    m = np.zeros(motion_dim_continuous(window), dtype=np.float32)
    if last_action is not None:
        m[0] = float(last_action[0])
        m[1] = float(last_action[1])
    m[2] = dx / step
    m[3] = dy / step
    if window > 1:
        m[4] = net_dx / (window * step)
        m[5] = net_dy / (window * step)
    return m


class MotionStateContinuous:
    """Per-rollout tracker, continuous-action analog of motion.MotionState.

    Usage each step, at robot pose (x, y):
        motion = ms.vec(x, y)
        action = policy.select_action(state_with(motion))
        ms.commit(x, y, action)
        env.step(action)
    """

    def __init__(self, window: int = 1):
        self.window = window
        self.reset()

    def reset(self):
        self.last_action = None
        self.prev = None
        self.history = deque(maxlen=max(1, self.window))

    def vec(self, x, y):
        if self.prev is None:
            dx = dy = 0.0
        else:
            dx, dy = x - self.prev[0], y - self.prev[1]
        if self.window > 1 and self.history:
            ox, oy = self.history[0]
            net_dx, net_dy = x - ox, y - oy
        else:
            net_dx = net_dy = 0.0
        return make_motion_continuous(self.last_action, dx, dy, net_dx, net_dy,
                                      window=self.window)

    def commit(self, x, y, action):
        self.history.append((x, y))
        self.prev = (x, y)
        self.last_action = action
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sac_motion.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add sac_motion.py tests/test_sac_motion.py
git commit -m "feat(sac): continuous-action motion features"
```

---

### Task 3: `sac_model.py` — Policy + Critic (ported from sac-fetch)

**Files:**
- Create: `sac_model.py`
- Test: `tests/test_sac_model.py`

**Interfaces:**
- Produces: `class Critic(nn.Module): __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='q_network')`, `.forward(state, action) -> (q1, q2)`; `class Policy(nn.Module): __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name='policy_network')`, `.forward(state) -> (mean, log_std)`, `.sample(state) -> (action, log_prob, mean)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sac_model.py
import numpy as np
import torch
import gymnasium as gym
from sac_model import Policy, Critic


def _action_space():
    return gym.spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                          high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)


def test_critic_forward_shapes():
    critic = Critic(num_inputs=6, num_actions=2, hidden_dim=128)
    state = torch.randn(8, 6)
    action = torch.randn(8, 2)
    q1, q2 = critic(state, action)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)


def test_policy_sample_shapes_and_action_bounds():
    policy = Policy(num_inputs=6, num_actions=2, hidden_dim=128, action_space=_action_space())
    state = torch.randn(8, 6)
    action, log_prob, mean = policy.sample(state)
    assert action.shape == (8, 2)
    assert log_prob.shape == (8, 1)
    assert mean.shape == (8, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_policy_sample_is_stochastic():
    policy = Policy(num_inputs=6, num_actions=2, hidden_dim=128, action_space=_action_space())
    state = torch.randn(1, 6)
    a1, _, _ = policy.sample(state)
    a2, _, _ = policy.sample(state)
    assert not torch.allclose(a1, a2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sac_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sac_model'`

- [ ] **Step 3: Write the implementation**

```python
# sac_model.py
"""Vanilla double-Q SAC actor/critic, ported from sac-fetch/model.py unchanged.

No image input, no goal-encoder branch -- the goal is already folded into the
flat state vector (ego_vector + motion) by sac_episode_buffer.py before it
ever reaches these networks.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='q_network'):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.output2 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.output1(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.output2(x2)

        return x1, x2

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                 checkpoint_dir='checkpoints', name='policy_network'):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sac_model.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add sac_model.py tests/test_sac_model.py
git commit -m "feat(sac): port Policy/Critic from sac-fetch"
```

---

### Task 4: `sac_buffer.py` — flat replay buffer

**Files:**
- Create: `sac_buffer.py`
- Test: `tests/test_sac_buffer.py`

**Interfaces:**
- Produces: `class SACReplayBuffer: __init__(self, max_size, state_dim, action_dim)`, `.can_sample(batch_size) -> bool`, `.store_transition(state, action, reward, next_state, done)`, `.sample_buffer(batch_size) -> (states, actions, rewards, next_states, dones)` (all `np.ndarray`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sac_buffer.py
import numpy as np
from sac_buffer import SACReplayBuffer


def test_can_sample_false_when_empty():
    buf = SACReplayBuffer(max_size=1000, state_dim=6, action_dim=2)
    assert buf.can_sample(batch_size=8) is False


def test_can_sample_true_after_enough_transitions():
    buf = SACReplayBuffer(max_size=1000, state_dim=6, action_dim=2)
    for _ in range(80):
        buf.store_transition(np.zeros(6), np.zeros(2), 0.0, np.zeros(6), False)
    assert buf.can_sample(batch_size=8) is True


def test_store_and_sample_roundtrip():
    buf = SACReplayBuffer(max_size=1000, state_dim=6, action_dim=2)
    state = np.arange(6, dtype=np.float32)
    action = np.array([0.5, -0.5], dtype=np.float32)
    buf.store_transition(state, action, 1.0, state * 2, True)

    states, actions, rewards, next_states, dones = buf.sample_buffer(batch_size=1)
    assert np.allclose(states[0], state)
    assert np.allclose(actions[0], action)
    assert rewards[0] == 1.0
    assert np.allclose(next_states[0], state * 2)
    assert dones[0] == True


def test_wraps_around_at_capacity():
    buf = SACReplayBuffer(max_size=4, state_dim=2, action_dim=1)
    for i in range(6):
        buf.store_transition(np.array([i, i], dtype=np.float32), np.array([i], dtype=np.float32),
                             float(i), np.array([i, i], dtype=np.float32), False)
    assert buf.mem_ctr == 6
    # index 6 % 4 == 2 was the last write -> slot 2 holds transition i=2 overwritten by i=6
    assert buf.state_memory[2][0] == 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sac_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sac_buffer'`

- [ ] **Step 3: Write the implementation**

```python
# sac_buffer.py
"""Flat replay buffer for continuous SAC. Same shape as sac-fetch's
buffer.py, trimmed of the Kitchen-specific expert-data/augmentation/CSV
extras this project doesn't use."""
import numpy as np


class SACReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_ctr = 0

        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        return (self.state_memory[batch], self.action_memory[batch],
                self.reward_memory[batch], self.next_state_memory[batch],
                self.terminal_memory[batch])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sac_buffer.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sac_buffer.py tests/test_sac_buffer.py
git commit -m "feat(sac): flat replay buffer"
```

---

### Task 5: `sac_episode_buffer.py` — HER relabeling with `ego_vector`

**Files:**
- Create: `sac_episode_buffer.py`
- Test: `tests/test_sac_episode_buffer.py`

**Interfaces:**
- Consumes: `goal_geometry.ego_vector(rx, ry, rtheta, gx, gy) -> np.ndarray(2,)` (existing); `sac_buffer.SACReplayBuffer.store_transition` (Task 4); a `compute_reward(achieved_goal_batch, desired_goal_batch, info) -> np.ndarray` callable shaped like `HomeBotGoalEnv.compute_reward`.
- Produces: `class SACEpisodeBuffer: K = 2`, `.store(action, reward, done, achieved_prev, achieved_next, heading_prev, heading_next, motion_prev, motion_next)`, `.__len__()`, `.clear()`, `.send_to(replay_buffer, desired_goal, compute_reward, k=None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sac_episode_buffer.py
import numpy as np
from sac_episode_buffer import SACEpisodeBuffer
from goal_geometry import ego_vector


class FakeReplayBuffer:
    """Captures store_transition calls for assertion instead of a real buffer."""
    def __init__(self):
        self.calls = []

    def store_transition(self, state, action, reward, next_state, done):
        self.calls.append((state, action, reward, next_state, done))


def fake_compute_reward(achieved, desired, info):
    diff = np.asarray(achieved, dtype=np.float32) - np.asarray(desired, dtype=np.float32)
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= 31.0).astype(np.float32)


def test_empty_buffer_sends_nothing():
    eb = SACEpisodeBuffer()
    rb = FakeReplayBuffer()
    eb.send_to(rb, desired_goal=np.array([0.0, 0.0]), compute_reward=fake_compute_reward)
    assert rb.calls == []


def test_single_transition_real_goal_state_matches_ego_vector():
    eb = SACEpisodeBuffer()
    motion = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    eb.store(
        action=np.array([1.0, 0.0]), reward=0.0, done=False,
        achieved_prev=np.array([100.0, 100.0]), achieved_next=np.array([104.0, 100.0]),
        heading_prev=0.0, heading_next=0.0,
        motion_prev=motion, motion_next=motion,
    )
    rb = FakeReplayBuffer()
    desired_goal = np.array([200.0, 100.0])
    eb.send_to(rb, desired_goal=desired_goal, compute_reward=fake_compute_reward, k=0)

    assert len(rb.calls) == 1
    state, action, reward, next_state, done = rb.calls[0]
    expected_goal = ego_vector(100.0, 100.0, 0.0, 200.0, 100.0)
    assert np.allclose(state[:2], expected_goal)
    assert np.allclose(state[2:], motion)
    assert reward == 0.0
    assert done is False


def test_hindsight_relabel_produces_terminal_success():
    eb = SACEpisodeBuffer()
    motion = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    # Step 0: far from any goal.
    eb.store(action=np.array([1.0, 0.0]), reward=0.0, done=False,
             achieved_prev=np.array([0.0, 0.0]), achieved_next=np.array([10.0, 0.0]),
             heading_prev=0.0, heading_next=0.0, motion_prev=motion, motion_next=motion)
    # Step 1: lands exactly on what step 0 will be relabeled toward.
    eb.store(action=np.array([1.0, 0.0]), reward=0.0, done=False,
             achieved_prev=np.array([10.0, 0.0]), achieved_next=np.array([20.0, 0.0]),
             heading_prev=0.0, heading_next=0.0, motion_prev=motion, motion_next=motion)

    rb = FakeReplayBuffer()
    # k=1 forces exactly one hindsight relabel per eligible transition.
    eb.send_to(rb, desired_goal=np.array([999.0, 999.0]), compute_reward=fake_compute_reward, k=1)

    # 2 original + 1 hindsight (only step 0 has a future transition to sample from).
    assert len(rb.calls) == 3
    hindsight_call = rb.calls[-1]
    _, _, reward, _, done = hindsight_call
    assert reward == 1.0  # achieved_next (10,0) == hindsight goal (20,0)'s achieved_next? see below
    assert done is True


def test_clear_empties_buffer():
    eb = SACEpisodeBuffer()
    eb.store(action=np.array([0.0, 0.0]), reward=0.0, done=False,
             achieved_prev=np.array([0.0, 0.0]), achieved_next=np.array([0.0, 0.0]),
             heading_prev=0.0, heading_next=0.0,
             motion_prev=np.zeros(4), motion_next=np.zeros(4))
    assert len(eb) == 1
    eb.clear()
    assert len(eb) == 0
```

**Note on `test_hindsight_relabel_produces_terminal_success`:** step 0's hindsight goal is sampled from `future = [step 1]`, so the only possible hindsight goal is step 1's `achieved_next = (20, 0)`. The relabel reward is `compute_reward(step0.achieved_next=(10,0), hindsight_goal=(20,0))` — distance is 10px, which is `<= 31.0`, so `reward == 1.0`. If this assertion fails, re-check `RELABEL_RADIUS`-equivalent threshold in the fake (31.0) against the actual distance in the test data before assuming the implementation is wrong.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sac_episode_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sac_episode_buffer'`

- [ ] **Step 3: Write the implementation**

```python
# sac_episode_buffer.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sac_episode_buffer.py -v`
Expected: 4 passed

If `test_hindsight_relabel_produces_terminal_success` fails on the reward value, print `hindsight_call` and re-derive the expected distance by hand before changing the implementation — the test docstring above shows the expected arithmetic.

- [ ] **Step 5: Commit**

```bash
git add sac_episode_buffer.py tests/test_sac_episode_buffer.py
git commit -m "feat(sac): HER episode buffer using ego_vector"
```

---

### Task 6: `sac_agent.py` — SACAgent (select_action, update_parameters, train loop)

**Files:**
- Create: `sac_agent.py`
- Test: `tests/test_sac_agent.py`

**Interfaces:**
- Consumes: `sac_model.Policy`, `sac_model.Critic` (Task 3); `sac_buffer.SACReplayBuffer` (Task 4); `sac_episode_buffer.SACEpisodeBuffer` (Task 5); `sac_motion.MotionStateContinuous` (Task 2); `goal_geometry.ego_vector` (existing).
- Produces: `class SACAgent: __init__(self, env, state_dim, action_dim, max_buffer_size=200000, hidden_dim=128, gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4, motion_window=1)`, `.select_action(state, evaluate=False) -> np.ndarray`, `.update_parameters(batch_size) -> (critic_loss: float, policy_loss: float, mean_q: float)`, `.train(episodes=1800, batch_size=64, run_tag=None)`, `.save_checkpoint()`, `.load_checkpoint(evaluate=False)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sac_agent.py
import numpy as np
import gymnasium as gym
import torch
from sac_agent import SACAgent


class FakeEnv:
    """Minimal stand-in -- only what SACAgent's constructor touches."""
    action_space = gym.spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                                  high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)


def test_select_action_within_bounds():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    state = np.random.randn(6).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_select_action_evaluate_is_deterministic():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    state = np.random.randn(6).astype(np.float32)
    a1 = agent.select_action(state, evaluate=True)
    a2 = agent.select_action(state, evaluate=True)
    assert np.allclose(a1, a2)  # evaluate path returns the policy mean, not a sample


def test_update_parameters_runs_without_nan_and_returns_three_floats():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    for _ in range(200):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, size=2).astype(np.float32)
        ns = np.random.randn(6).astype(np.float32)
        agent.memory.store_transition(s, a, float(np.random.rand()), ns, False)

    critic_loss, policy_loss, mean_q = agent.update_parameters(batch_size=16)
    assert np.isfinite(critic_loss)
    assert np.isfinite(policy_loss)
    assert np.isfinite(mean_q)


def test_update_parameters_changes_policy_weights():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    for _ in range(200):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, size=2).astype(np.float32)
        ns = np.random.randn(6).astype(np.float32)
        agent.memory.store_transition(s, a, float(np.random.rand()), ns, False)

    before = agent.policy.linear1.weight.clone().detach()
    agent.update_parameters(batch_size=16)
    after = agent.policy.linear1.weight.clone().detach()
    assert not torch.allclose(before, after)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sac_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sac_agent'`

- [ ] **Step 3: Write the implementation**

```python
# sac_agent.py
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

    def train(self, episodes=1800, batch_size=64, run_tag=None):
        run_tag = run_tag or self._run_tag()
        writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sac_agent.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sac_agent.py tests/test_sac_agent.py
git commit -m "feat(sac): SACAgent with HER training loop"
```

---

### Task 7: `train_sac.py` + end-to-end smoke test against the real env

**Files:**
- Create: `train_sac.py`
- Test: `tests/test_sac_smoke.py`

**Interfaces:**
- Consumes: `sac_agent.SACAgent` (Task 6); `gym.make("HomeBot2D-Goal-V1", ...)` (real env, Task 1's pin).
- Produces: a runnable training entry point; no new interfaces for later tasks (this is the integration point).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_sac_smoke.py
"""End-to-end smoke test: real env, a handful of episodes, must not crash
or produce NaNs. This is NOT a convergence test -- it's the pipeline-runs
check from the design spec, mirroring this repo's existing HER smoke tests."""
import math

import gymnasium as gym
import homebot  # noqa: F401  (registers HomeBot2D-Goal-V1)
import numpy as np

from sac_agent import SACAgent
from sac_motion import motion_dim_continuous
from goal_geometry import ego_vector


def _make_env():
    return gym.make(
        "HomeBot2D-Goal-V1",
        render_mode=None,
        action_mode="continuous",
        obs_resolution=(96, 96),
        n_trash=1,
        max_steps=50,  # short episodes -- this test just needs the pipe to run
        map_name="default",
        goals=["collect_trash"],
        random_start=True,
    )


def test_smoke_few_episodes_no_nan_no_crash():
    env = _make_env()
    state_dim = 2 + motion_dim_continuous(window=1)  # ego_vector(2) + motion(4)
    agent = SACAgent(env=env, state_dim=state_dim, action_dim=2,
                     max_buffer_size=5000, hidden_dim=32)

    agent.train(episodes=5, batch_size=16, run_tag="smoke-test")

    assert agent.total_env_steps > 0
    assert math.isfinite(agent.policy.linear1.weight.sum().item())
    assert math.isfinite(agent.critic.linear1.weight.sum().item())
    env.close()


def test_smoke_state_dim_matches_what_build_state_produces():
    env = _make_env()
    state_dim = 2 + motion_dim_continuous(window=1)
    agent = SACAgent(env=env, state_dim=state_dim, action_dim=2,
                     max_buffer_size=5000, hidden_dim=32)

    raw_obs, _ = env.reset()
    base = env.unwrapped
    r = base._robot
    motion = np.zeros(motion_dim_continuous(window=1), dtype=np.float32)
    state = agent._build_state(r.x, r.y, r.angle,
                               raw_obs["desired_goal"][0], raw_obs["desired_goal"][1], motion)
    assert state.shape == (state_dim,)
    env.close()
```

- [ ] **Step 2: Run the smoke test to verify it fails (or passes for the wrong reason)**

Run: `pytest tests/test_sac_smoke.py -v`
Expected at this point: should already PASS, since Tasks 1-6 are complete — this step is really "run it once before writing `train_sac.py` to confirm the agent alone is sufficient," not a red/green TDD gate. If it fails, the failure is in Tasks 1-6, not this task — stop and fix there first.

- [ ] **Step 3: Write `train_sac.py`**

```python
# train_sac.py
"""Continuous SAC + HER on the collect_trash leg. Stability-first v1: no
image observation, no n-step, no macro-actions, no goal noise. Mirrors
train.py's collect_trash config (n_trash=1, random_start=True) so this is
a clean A/B against the discrete champion's reference run, modulo algorithm."""
import gymnasium as gym
import homebot  # noqa: F401

from sac_agent import SACAgent
from sac_motion import motion_dim_continuous

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="continuous",
    obs_resolution=(96, 96),
    n_trash=1,
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,
)

STATE_DIM = 2 + motion_dim_continuous(window=1)  # ego_vector(2) + motion(4)

agent = SACAgent(env=env, state_dim=STATE_DIM, action_dim=2,
                 max_buffer_size=200000, hidden_dim=128,
                 gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4, motion_window=1)

agent.train(episodes=1800, batch_size=64)
```

- [ ] **Step 4: Run the smoke test again to verify everything still passes**

Run: `pytest tests/test_sac_smoke.py tests/test_sac_motion.py tests/test_sac_model.py tests/test_sac_buffer.py tests/test_sac_episode_buffer.py tests/test_sac_agent.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add train_sac.py tests/test_sac_smoke.py
git commit -m "feat(sac): train_sac.py entry point + end-to-end smoke test"
```

---

### Task 8: Push and launch in Beekeeper (not TDD — operational)

**Files:** none (deployment step)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin sac-continuous-her
```

- [ ] **Step 2: Check Beekeeper project config before starting training**

Use `mcp__beekeeper__get_project` / `get_project_instructions` for the sac-homebot project to confirm how it expects to be pointed at a branch and what the pre-launch pip-install step does (relevant since Task 1 changed `requirements.txt`).

- [ ] **Step 3: Start training**

Use `mcp__beekeeper__start_training`, targeting branch `sac-continuous-her`, entry point `train_sac.py`.

- [ ] **Step 4: Watch the first several hundred episodes for the specific failure mode this plan is testing for**

Use `mcp__beekeeper__analyze_run` / TensorBoard, watching `Train/mean_q` specifically (per `CLAUDE.md`: raw logs are for crash/error detection only, not trend analysis — a noisy `Train/episode_reward` tail is expected and not a signal). A `mean_q` that grows without bound (vs. settling) is the literal overestimation-collapse signature from `goal_reacher_overestimation.md` — that is the one thing v1 is explicitly trying to avoid, and the only metric that should change this plan's verdict.

---

## Self-Review

**Spec coverage:**
- Repo layout / what's ported vs. left behind → Task 3 (engine port), constraints section (explicit exclusions).
- Observation & action interface (`ego_vector`, continuous action space, no CNN) → Tasks 2, 5, 6, 7.
- HER / replay buffer / reward, `_blocked_penalty` exclusion, no n-step → Tasks 4, 5.
- Training scope (`collect_trash` only, mirrors `train.py`) → Task 7.
- TensorBoard run-tagging, mean-Q instrumentation → Task 6.
- Eval integration open item → explicitly NOT in this plan's scope; flagged in the spec as a follow-up once training is validated, not blocking v1.
- `requirements.txt` staleness fix → Task 1.

**Placeholder scan:** no TBD/TODO; every code step is complete, runnable code.

**Type consistency check:** `state_dim` is computed identically in three places (`tests/test_sac_smoke.py`, `train_sac.py`) as `2 + motion_dim_continuous(window=1)` = 6 — consistent. `SACAgent.__init__` signature matches its usage in Tasks 6 and 7 exactly (`env, state_dim, action_dim, max_buffer_size, hidden_dim, gamma, tau, alpha, lr, motion_window`). `SACEpisodeBuffer.send_to`'s `compute_reward` parameter is called the same way in Task 5's tests and Task 6's `train()` (`base.compute_reward`, signature `(achieved, desired, info)`).
