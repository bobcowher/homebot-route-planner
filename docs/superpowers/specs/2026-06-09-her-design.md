# HER Implementation Design

**Branch:** `her`
**Date:** 2026-06-09

## Overview

Implement Hindsight Experience Replay (HER) with a goal-conditioned DQN on the `HomeBot2D-Goal-v1` environment. The environment already exposes `achieved_goal` (robot pixel coords) and `desired_goal` (target pixel coords) in every obs dict, and provides a batched `compute_reward` for relabeling.

---

## Components

### 1. `models/q_model.py` — Goal-conditioned QModel

Add a `goal_encoder` branch. The CNN processes the image; the goal encoder processes the `desired_goal`. Both outputs are concatenated before the FC head.

```
CNN(obs [3,96,96])  →  flat [4096]
goal [2]  →  Linear(2→128)  →  ReLU  →  goal_embed [128]
cat([flat, goal_embed]) [4224]  →  FC(4224→512)  →  ReLU  →  Linear(512→8)
```

- Constructor gains `goal_dim: int = 2`
- `forward(obs, goal)` — both tensors required; goal is raw pixel coords (no external normalization; the encoder learns to scale)
- Goal coords are in pixel space (~0–800 range for default map)

### 2. `buffer.py` — Goal-aware ReplayBuffer

Add a `goal_memory` tensor `(max_size, goal_dim)` float32 on `input_device`.

- `store_transition(state, action, reward, next_state, done, goal)` — `goal` is a `(2,)` float32 numpy array
- `sample_buffer(batch_size)` returns a 6-tuple: `states, actions, rewards, next_states, dones, goals`

### 3. `episode_buffer.py` — HER relabeling in `flush_to`

`flush_to(replay_buffer, desired_goal, compute_reward)` does two passes:

**Pass 1 — original transitions:**
Store each transition with `desired_goal` (the episode's actual goal) and the env-computed reward as-is.

**Pass 2 — hindsight transitions (K=4, future strategy):**
For each step `i`:
- `future = self._transitions[i + 1:]`
- If `future` is empty, skip (last step has no future states to sample from)
- Sample `min(K, len(future))` goals without replacement from `future[j].achieved_goal`
- For each sampled goal: call `compute_reward(t.achieved_goal[np.newaxis], hindsight_goal[np.newaxis], {})` → float reward
- Store relabeled transition with `hindsight_goal` as the goal

Result: up to 5× buffer density per episode (1 original + up to 4 hindsight per step).

### 4. `agent.py` — Threading goal through training

**`select_action(obs, goal)`:**
- Epsilon-greedy unchanged
- Greedy path: `q_model(obs_t, goal_t).argmax()`
- `goal_t = torch.as_tensor(goal, dtype=torch.float32, device).unsqueeze(0)`

**`train_step(batch_size)`:**
- Unpack 6-tuple from `sample_buffer`
- Online net: `q_model(obs, goals).gather(1, actions)`
- Target net: `target_q_model(next_obs, goals).gather(1, next_actions)`

**Training loop (`train`):**
- Capture `desired_goal = raw_obs["desired_goal"]` at episode reset — stable for the full episode
- Pass `desired_goal` to `select_action` at every step
- Pass `desired_goal` + `env.unwrapped.compute_reward` to `flush_to` at episode end

---

## Data Flow

```
env.reset() → raw_obs dict
  └─ obs = process(raw_obs["observation"])
  └─ desired_goal = raw_obs["desired_goal"]  ← fixed for episode

per step:
  select_action(obs, desired_goal) → action
  env.step(action) → raw_next
    └─ next_obs = process(raw_next["observation"])
    └─ achieved_goal = raw_next["achieved_goal"]
  episode_buffer.store(..., achieved_goal)

end of episode:
  flush_to(memory, desired_goal, compute_reward)
    ├─ original transitions  → memory  (goal = desired_goal)
    └─ hindsight transitions → memory  (goal = sampled future achieved_goal)
  episode_buffer.clear()

train_step:
  sample_buffer → (obs, actions, rewards, next_obs, dones, goals)
  q_model(obs, goals) / target_q_model(next_obs, goals)
```

---

## Constraints

- Buffer capacity stays at 100k. At 1000 steps/episode and K=4, a full episode writes ~5k transitions — buffer holds ~20 full episodes worth of data.
- No goal normalization in this implementation. The `goal_encoder` learns to scale pixel coords.
- HER strategy is `future` only. `final` and `episode` strategies are not implemented.
- `test()` also passes `desired_goal` to `select_action` — goal is available from `raw_obs["desired_goal"]` at reset.

---

## Files Changed

| File | Change |
|------|--------|
| `models/q_model.py` | Add `goal_dim`, `goal_encoder`, update `forward` signature |
| `buffer.py` | Add `goal_memory`, update `store_transition` and `sample_buffer` |
| `episode_buffer.py` | Implement HER relabeling loop in `flush_to` |
| `agent.py` | Thread `goal` through `select_action`, `train_step`, `train`, `test` |
