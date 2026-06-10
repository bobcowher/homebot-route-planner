# Q-Homebot Tuning Log

Overnight autonomous tuning session. Goal: get HER + Double-DQN agent on
`HomeBot2D-Goal-v1` (`collect_trash`, n_trash=2, max_steps=1000) to solve the
task **consistently** (env max reward = 1.0 per episode → success = consistent 1s).

Method: change one variable at a time. Kick off a Beekeeper run with a clear
TensorBoard tag, watch it, compare to baseline. If it doesn't beat baseline,
**roll it back** before trying the next lever. Each run is tagged via the
remote branch name, so tuning variants live on dedicated `tuning-*` branches
(or are noted here when the tag is reused).

Single GPU, parallel runs disabled → one run at a time. Baseline TB data is
retained (tb_logs_max_runs=10) for comparison.

**Branch hygiene:** each experiment gets a `tuning-*` branch (drives the TB
tag). When an experiment concludes — loser: roll code back + delete local and
remote branch; winner: fold into `tuning` + delete the experiment branch. Goal
is at most one or two live `tuning-*` branches, not a graveyard. Cleanup status
tracked per experiment below.

### Branch cleanup ledger
- `tuning-huber` (run 224): live — pending Exp 1 verdict.

---

## Baseline — Run 223 (`her` branch)

Config: lr=1e-4, MSE loss, 800 grad-steps/episode, batch=64, gamma=0.99,
epsilon 1.0→0.1 decay 0.977 (min at ep ~100), hard target update every 1000
steps, HER K=4 (future strategy).

Result after ~530 episodes:
- `Train/best_score` = **1.0** — architecture is sound, the goal is reachable.
- `Train/episode_reward` smoothed ≈ **0.26**, peak ≈ 0.47 — ~1-in-4 success.
- Successes are fast (1–120 steps); failures burn the full 1000 steps (reward 0).
- `Train/avg_q_loss` **worsening / unstable** — spikes to 2000–4600 despite
  grad-norm clip at 1.0. Q-values are diverging. Suspected cap on the policy:
  exploding Q estimates → noisy argmax → inconsistent success.

**Baseline number to beat: smoothed episode_reward ≈ 0.26.**

---

## Experiments

### Exp 1 — Huber (smooth_l1) loss instead of MSE
- **Hypothesis:** MSE squares large TD errors, producing the 2000–4600 loss
  spikes and Q divergence. Huber is linear past delta=1, the textbook DQN fix
  for exactly this symptom. Should stabilize Q and lift/steady success rate.
- **Change:** `agent.py` train_step — `F.mse_loss` → `F.smooth_l1_loss`.
- **Tag/branch:** `tuning-huber` (TB tag derived from branch name).
- **Run:** 224 (baseline 223 stopped to free the GPU; its TB data retained).
- **Status:** RUNNING — awaiting comparison vs baseline 0.26.
