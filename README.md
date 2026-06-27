# homebot-route-planner (discrete SAC)

Discrete **Soft Actor-Critic + Hindsight Experience Replay (HER)** for goal-conditioned
navigation in the `HomeBot2D-Goal-V1` environment: a robot must drive to a commanded goal
(the trash pile) on a 2D home map, from a random start, using only an egocentric image and
a noisy goal vector.

> The pure Q-learning champion that preceded this lives in its own repo,
> [`q-homebot-route-planner`](https://github.com/bobcowher/q-homebot-route-planner). This
> repo is SAC-only.

## The recipe

```
4x512 double-Q critic + categorical actor   (model.py)
HER, relabel-to-achieved                     (episode_buffer.py)  ← this is the curriculum
fixed temperature alpha = 0.1                 (no auto-tuning)
behaviour = epsilon-greedy ARGMAX over the   (agent.greedy_critic_action)
            critic's min-Q, epsilon 1.0->0.1
max_steps = 1000, gamma = 0.99, warmup 5000  (train.py)
```

On the full random-start map this reaches the goal ~20% of episodes and sustains it —
including far spawns that need most of the 1000-step budget to traverse.

## Run

```bash
./build.sh            # activates the conda env and runs train.py
# or: python -u train.py
```

TensorBoard scalars (`runs/`): `Train/episode_reward`, `Train/episode_steps`,
`Train/mean_q`, `Train/policy_entropy`, `Train/epsilon`, `loss/critic`, `loss/actor`.

## Layout

| file | role |
|------|------|
| `agent.py` | `SACAgent` — the discrete SAC update, HER wiring, epsilon-greedy critic-greedy rollout, training loop |
| `model.py` | `DiscreteQNet` (double-Q critic) + `DiscretePolicy` (categorical actor), shared `_CNNBase` |
| `buffer.py` | image-aware replay buffer (uint8 frames, int actions) |
| `episode_buffer.py` | per-episode cache + HER relabeling (future strategy) |
| `motion.py` | `MotionStateDiscrete` anti-oscillation feature |
| `goal_geometry.py` | goal vector, robot/step geometry |
| `train.py` | entry point + hyperparameters |
| `tests/` | unit + end-to-end smoke tests |

## Why these choices (the non-obvious findings)

These took a long search to pin down; they are *not* the textbook defaults.

- **HER is the curriculum.** Relabeling to achieved goals trains the agent on goals it
  actually reached (easy-first, auto-shrinking). No spawn/reach curriculum is needed — that
  was a crutch for an under-capacity critic.
- **Behaviour comes from the *critic*, not the actor.** The HER critic learns a good value
  field, but the inter-action *advantage* (how much one action helps toward the goal) is
  small relative to the state-value baseline. The entropy-regularised actor washes that
  small signal out and stays near-uniform; `argmax` over the critic's min-Q follows it
  regardless of magnitude — the same mechanism that makes DQN work here. The actor is kept
  only because it feeds the soft-value bootstrap.
- **Fixed alpha, no auto-tuning.** Automatic entropy tuning never converged usefully on this
  task (it oscillated, collapsed, or went inert); a static `alpha = 0.1` is reliable.
- **Capacity over regularization for stability.** A 4x512 critic is stable; a 2x256 critic
  diverges. LayerNorm bounds the critic but normalizes away the small Q-spread behaviour
  depends on, so it's avoided.
- **Long episodes (`max_steps=1000`).** The discrete robot moves 4px/step, so short episodes
  can't traverse the map; long trajectories give HER rich relabel data and let the agent
  reach far goals.
