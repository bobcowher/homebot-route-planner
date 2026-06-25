# Continuous SAC + HER Re-baseline — Design

**Branch:** `sac-continuous-her` (off `noisy-goal-vector`, i.e. `main` + the latest goal-vector-noise commit)
**Date:** 2026-06-24

## Overview

Re-baseline SAC for HomeBot navigation using `action_mode="continuous"` (`HomeBotGoalEnv`'s `Box([-1,1]^2)` = `[linear, angular]`, heading-relative — confirmed empirically, see Constraints). The original SAC line on this project died to value overestimation on a polar PointGoal reacher (`goal_reacher_overestimation.md`); this re-baseline carries forward everything learned since (HER, reward-radius fix, goal-vector framing) onto a clean, vanilla double-Q SAC engine.

**Success bar (v1): stability, not parity.** Train cleanly on heading control without the overestimation collapse. Matching champion-314/run325's reach %, spin %, or chain score is explicitly not a v1 gate — that's a later concern once the engine is proven to converge at all.

**Engine source:** `sac-fetch` (`sac-farama-kitchen`, `~/pythonprojects/sac-fetch`) — `model.py` (tanh-Gaussian `Policy` + double-Q `Critic`) and `agent.py` (`Agent.select_action` / `update_parameters`), ported as-is. Fixed alpha (no auto-entropy-tuning machinery present) already matches this project's standing preference (`sac_fixed_alpha.md`).

**Explicitly not ported** from `sac-fetch`: `meta_agent.py`, `train_meta.py`, `test_meta.py`, `controller.py`, `human_control.py`, pretrain/curiosity scaffolding. That's Kitchen-specific multi-task/teleop infra (one network per skill); HomeBot goal-conditioning is one shared network with the goal in the state, trained via HER, which works completely differently.

**Also not reused:** the discrete-SAC entropy-tuning code archived on `sac-discrete-explore` (different action space, an already-rejected direction per `sac_fixed_alpha.md` / `soft_q_training_verdict.md`).

---

## Environment (ground truth, `gym-homebot-2d`)

Verified against `gym-homebot-2d@bf0dba6` (pushed to `origin/main`) — **not** the stale `de9c437` currently pinned in this repo's `requirements.txt` (see Constraints).

- Env id: `HomeBot2D-Goal-V1` → `homebot.env.HomeBotGoalEnv`. Dict observation space: `{"observation": Box(0,255,(h,w,3)), "achieved_goal": Box(2,), "desired_goal": Box(2,)}`.
- One active goal per episode, chosen at `reset(options={"goal": name})`. `GOAL_REGISTRY` (in `homebot/goals.py`): `go_to_fridge`, `deliver_drink`, `go_to_door`, `deliver_package`, `collect_trash` — each maps to a fixture/trash target and an optional pre-loaded carry state (`deliver_*` goals pre-load the carried item during training; `evaluate=True` skips the pre-load).
- Reward: `step()` uses `TaskManager.step(robot)` — the real per-target reward, firing at the real interaction radius (trash 31px / door 47px / fixture 79px). `terminated = reward > 0.5`. `compute_reward(achieved, desired, info)` is now **only** the HER hindsight-relabel proxy: sparse 0/1 at a single tight `RELABEL_RADIUS=31`. This is the run325 fix (`reward_radius_bug_run325.md`) — real transitions and HER-relabeled transitions use different reward paths, by design.
- `action_mode="continuous"` → `Box([-1,1]^2)` = `[linear, angular]`, identical interface on both `HomeBotEnv` and `HomeBotGoalEnv`. Confirmed empirically (see Constraints) that movement is heading-relative: turning accumulates into `robot.angle`, and "forward" moves along whatever heading currently is, not a fixed world axis.

## Observation & Action Interface

- Action: continuous `[linear, angular]` straight to `env.step()`. No decode/encode layer.
- Core goal-relative feature: `goal_geometry.ego_vector(rx, ry, robot.angle, gx, gy)` — already exists, already unit-tested. Rotates the world-frame goal displacement by `-robot.angle`, returning `[forward-distance-to-goal, left-distance-to-goal]`. This matches the action space's own frame (turn/forward are heading-relative), unlike `world_vector`/`noisy_world_vector` which are world-frame and were only correct for the discrete-8 (absolute-direction) action space.
- No separate raw-heading input needed — `ego_vector` already bakes the rotation in; the policy doesn't need to know its absolute heading, only the goal's position relative to its current facing.
- **No image/CNN input for v1.** The champion DQN+HER uses a CNN over the rendered viewport concatenated with the goal vector; `sac-fetch`'s `Policy`/`Critic` are pure MLP. Reusing the engine "as-is" plus the stability-first bar both argue against adding a CNN here — more parameters and a second input modality work against stabilizing a harder (heading) control problem on the first pass. **Known, deliberate gap:** this drops the only source of wall/fixture visual context the current system has (`ego_vector`, like `world_vector`, carries no obstacle information). Collisions still degrade gracefully (the env slides/blocks, no penalty), so this is a capability reduction, not a correctness bug — but if training shows the agent repeatedly driving into walls/fixtures, re-adding a CNN branch is the fix, not patching around it with more vector features. Fast-follow, not a silent omission.
- Noisy variant (`noisy_ego_vector`, mirroring today's `noisy_world_vector` memorization-breaking trick) is deferred — not required for the stability bar.
- `noisy_ego_vector` aside: nothing else changes in the per-step observation pipeline.

## HER, Replay Buffer, Reward

- New continuous-action episode buffer, adapting `episode_buffer.py`'s relabeling logic: same K-future hindsight strategy and reward-radius-curriculum hook (`reach_radius_at`/`reach_reward` from `goal_geometry.py`, already radius-parametrized and action-space-agnostic — unchanged).
- **`_blocked_penalty` does not port as-is.** Its logic ("near-zero displacement == a wall pin") assumes every action is a real commanded move, true for discrete-8 but false for continuous `[linear, angular]` — zero displacement is also the correct outcome of a legitimate "stand still" action (`linear≈0`). Dropped for v1. Revisit only if wall-sticking shows up as an observed problem during training, and if so, gate it on commanded linear magnitude (e.g. `moved < eps AND |linear| > threshold`), not displacement alone.
- Relabeled-goal reward: `goal_geometry.reach_reward(achieved, desired, radius)`, unchanged.
- Real (non-relabeled) transitions: the actual `TaskManager` reward + `reward > 0.5` termination, exactly as `HomeBotGoalEnv.step()` already provides — no separate reward logic to write.
- Target computation: plain 1-step SAC bootstrap (`r + γ·mask·(min(Q1,Q2) − α·logπ)`), straight out of `sac-fetch`'s `update_parameters`. **No n-step for v1** — n-step's boundary-crossing semantics were a real, previously-bitten bug class here, tied to DQN's max-Q target; SAC's entropy-adjusted target needs its own derivation, and the stability-first bar argues against adding that risk now.
- Motion/anti-oscillation feature (`motion.py`): the windowed net-displacement term (position-only) ports unchanged — it's still a relevant spin signal for a heading-controlled policy. The "last action" component does **not** port as a one-hot (that's sized by discrete `action_dim`) — it becomes the raw `[linear, angular]` float vector from the previous step instead.
- Flat replay buffer otherwise follows `sac-fetch`'s `buffer.py` shape: float actions, goal-vector storage analogous to this repo's existing goal-aware `buffer.py` (`goal_memory`/`next_goal_memory`, since the goal vector changes within a transition as the robot moves).

## Training Scope

- v1 mirrors today's `train.py` exactly: `goals=["collect_trash"], n_trash=1, random_start=True` — single leg, matching both the most current reference experiment and the documented dominant bottleneck (`collect_trash_root_cause.md`). The full 5-goal `GOAL_REGISTRY` pool is a fast-follow once this is validated, same as how the DQN+HER line iterated.
- Training stays scoped to single-goal episodes, full stop. Chaining (sequencing multiple goals) is **not** rebuilt for SAC — once a working single-goal policy exists, the existing `NavigatorTool`/`chained_eval.py` harness drives it unchanged to get chain numbers, the same way it already does for the discrete champion.
- New entry point (e.g. `train_sac.py`): `gym.make("HomeBot2D-Goal-V1", action_mode="continuous", ...)` + ported `Agent` + the new HER episode buffer + the existing goal-sampling/reset pattern (`evaluate=False` during training so `deliver_*`-style pre-loaded-carry goals behave correctly, matching `HomeBotGoalEnv`'s existing `evaluate` flag).
- TensorBoard: reuse this repo's branch-derived run-tagging (per `CLAUDE.md`) so runs tag correctly instead of showing up as `main`/`noisy-goal-vector`.
- Scalars: `loss/critic`, `loss/policy`, `reward/train` as `sac-fetch` already logs, plus a **mean/max-Q scalar** — the literal failure signature from `goal_reacher_overestimation.md` — so an overestimation blowup is visible from the first run, not discovered after the fact.

## Eval Integration (open item)

`evaluate.py`/`chained_eval.py` currently assume a discrete-Q-network-shaped agent interface. Not yet verified whether they're generic enough to take the new continuous `Agent.select_action` unchanged or need an adapter/new eval entry point — this is the first thing to check once the new `Agent` exists, not assumed to "just work."

`goal_geometry.spin_fraction` (purely position-trace-based) is reusable unchanged — still relevant, since a heading-based controller can oscillate too.

## Testing

- Smoke test mirroring this repo's existing HER-integration-smoke pattern: env reset → a few steps under the (randomly-initialized) policy → one episode pushed through HER relabeling → one `update_parameters()` call. Verifies the pipeline runs end-to-end without crashing, written before the full implementation (TDD).
- The `test_sac.py`/`smoke_sac.py` sitting on the archived `sac-discrete-explore` branch is not reusable — different action space, dead-end direction.
- No new bespoke error handling — inherit `sac-fetch`'s existing checkpoint try/except; the env already validates `action_mode`.

## Constraints

- **`requirements.txt` is currently pinned one commit behind the run325 reward fix** (`de9c437`, missing `bf0dba6`'s real-TaskManager-reward change — the commit message even says "bump version to force a clean pip reinstall," and was never picked up here). Bumping this pin to `bf0dba6` is part of this work, independent of SAC — without it, training would silently run against the flat-79px reward/termination bug this project already found and fixed once.
- Continuous-mode heading-relative physics confirmed empirically (not just by reading code): 5 steps of pure turn (`[0,1]`) advanced `robot.angle` 0.0→0.40 rad; a subsequent pure-forward step (`[1,0]`) produced `dx=3.68, dy=1.56`, matching `cos(0.4)*4, sin(0.4)*4` exactly. A no-turn forward control produced `dx=4.0, dy=0.0`. Robot.py is otherwise identical between the installed pip snapshot and the `bf0dba6` dev source (one cosmetic variable rename).
- Buffer capacity / HER ratios follow this repo's existing defaults (`max_buffer_size=200000`, K from `EpisodeBuffer`) unless training shows a reason to change them.

## Files Changed (new branch `sac-continuous-her`)

| File | Change |
|------|--------|
| `requirements.txt` | Bump `gym-homebot-2d` pin from `de9c437` to `bf0dba6` |
| `models/sac_model.py` (new) | Ported `Policy` + `Critic` from `sac-fetch/model.py`, unchanged |
| `sac_agent.py` (new) | Ported `Agent` core (`select_action`, `update_parameters`, checkpointing) from `sac-fetch/agent.py`; training-loop driver rewritten for HER + HomeBot |
| `sac_buffer.py` (new) | Flat replay buffer, `sac-fetch/buffer.py` shape, float actions + goal-vector storage matching this repo's existing `buffer.py` goal-memory pattern |
| `sac_episode_buffer.py` (new) | HER relabeling adapted from `episode_buffer.py`: float actions, `ego_vector` instead of `noisy_world_vector`, `_blocked_penalty` dropped |
| `motion.py` | Adapt `make_motion`/`motion_dim`/`MotionState`: last-action one-hot → raw `[linear, angular]` float vector |
| `goal_geometry.py` | No changes — `ego_vector`, `reach_reward`, `reach_radius_at`, `spin_fraction` all reused as-is |
| `train_sac.py` (new) | Entry point mirroring `train.py`'s `collect_trash`-only config, continuous `action_mode` |
| `tests/test_sac_smoke.py` (new) | End-to-end smoke test, written first |
