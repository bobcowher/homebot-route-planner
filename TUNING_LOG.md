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
- `tuning-huber` (run 224): WON → merged into `tuning`, branch deleted.
- `tuning-gamma` (run 225): NEUTRAL → reverted, branch deleted (not merged).
- `tuning-herk` (run 226): REGRESSION → reverted, branch deleted (not merged).
- `tuning-polyak` (run 227): REGRESSION → reverted, branch deleted (not merged).
- `tuning-lr` (run 228, LR 5e-5): WON → merged into `tuning`, branch deleted.
- `tuning-lr3e5` (run 229, LR 3e-5): REGRESSION → reverted, branch deleted.
- `tuning-batch` (run 230, batch 128): NEUTRAL → reverted, branch deleted.
- `tuning-episodes` (run 231, 2500 ep): NO BENEFIT → reverted, branch deleted.

All `tuning-*` experiment branches deleted; only `tuning` remains (best config).

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
  Completed full 1000 episodes in 33 min.
- **Result vs baseline:**
  - Q-loss smoothed **69 → 3.15**, spikes **4600 → 186**. Divergence solved.
  - episode_reward peak **0.47 → 0.57**; actual recent success rate ~30–33%
    (counted from log tail) vs baseline ~26%. (EMA `smoothed_final`=0.147 was
    depressed by a cold streak in the last 9 episodes — ignore it; use the tail
    count and peak.)
- **Verdict: WIN — KEEP.** Folded into `tuning`. Huber stabilizes Q and modestly
  lifts success. Stability headroom now enables raising gamma (Exp 2).

### Exp 2 — gamma 0.99 → 0.995 (long-horizon value propagation)
- **Hypothesis:** failures always burn all 1000 steps because distant goals are
  invisible: `0.99^1000 ≈ 4e-5`. Raising gamma to 0.995 (`0.995^200=0.37` vs
  `0.99^200=0.13`) propagates value further so the agent can "see" far trash.
  Huber + grad-clip should contain the larger targets.
- **Change:** `agent.py` `self.gamma = 0.99` → `0.995`. (On top of Huber.)
- **Tag/branch:** `tuning-gamma`.
- **Run:** 225 — completed full 1000 episodes (34 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~34–35% (tail count) vs ~32%; peak episode_reward 0.61 vs 0.57.
    Marginal, within episode-to-episode noise.
  - Q-loss smoothed 3.15 → **5.23**, spikes 186 → **369** — noisier (expected:
    larger targets), still far better than pre-Huber baseline.
  - Dominant failure mode unchanged: ~2/3 of episodes still burn all 1000 steps.
    Gamma alone did not crack long-horizon navigation.
- **Verdict: NEUTRAL → REVERT.** Gain within noise, costs Q-stability and muddies
  the story. Rolled back to gamma 0.99 (clean Huber baseline on `tuning`).

### Exp 3 — HER K 4 → 8 (denser hindsight signal for navigation)
- **Hypothesis:** the failure mode is reaching *distant* goals. HER's whole job is
  to teach "how to get to arbitrary positions" by relabeling. More hindsight goals
  per transition (K 8 vs 4) doubles the relabeled navigation signal per episode,
  which should lift the success rate where gamma couldn't. Single HER variable.
- **Change:** `episode_buffer.py` `K = 4` → `K = 8`. (On clean Huber baseline.)
- **Tag/branch:** `tuning-herk`.
- **Run:** 226 — completed full 1000 episodes (38 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~21% (tail count) vs ~32%; peak episode_reward 0.45 vs 0.57.
  - Longer cold streaks (e.g. ~28 consecutive failures, ep 881–909).
  - Q-loss smoothed 3.15 → 8.70 (worse).
- **Verdict: REGRESSION → REVERT.** Doubling relabels floods the buffer with
  easy near-future goals, shrinking the share of real desired-goal transitions →
  policy overfits trivial nearby goals, generalizes worse to distant trash.
  Rolled back to K=4 (clean Huber baseline on `tuning`).

### Exp 4 — Polyak soft target updates (replace hard update every 1000 steps)
- **Hypothesis:** three levers in, the wall is policy *instability across
  episodes* (cold streaks), not value reach or signal density. Hard target sync
  every 1000 grad-steps (≈ once/episode with 800/ep) makes the bootstrap target
  lurch once per episode. Polyak (tau=0.005, every step) tracks the online net
  smoothly → steadier targets → fewer cold streaks, higher floor. Pairs with the
  Huber stability story.
- **Change:** `agent.py` — add `self.tau = 0.005`; in train_step replace the
  `total_steps % target_update_interval` hard copy with a per-step soft update
  `tp = (1-tau)*tp + tau*p`. (On clean Huber baseline, K=4, gamma 0.99.)
- **Tag/branch:** `tuning-polyak`.
- **Run:** 227 — completed full 1000 episodes (36 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~17–18% (tail count) vs ~32%; peak episode_reward 0.49 vs 0.57.
  - Q-loss *noisier*: smoothed 6.68, spikes to 502 (vs 3.15 / 186).
  - Severe ~30-episode dry patch (ep 843–874).
- **Verdict: REGRESSION → REVERT.** With 800 grad-steps/episode, tau=0.005 every
  step tracks the online net ~98%/episode — *tighter* coupling than the hard sync,
  so it amplified the target-chasing feedback instead of damping it. Rolled back to
  hard target update. Lesson: this system is over-aggressive on updates.

### Exp 5 — learning rate 1e-4 → 5e-5 (calm the over-aggressive updates)
- **Hypothesis:** Polyak made things worse by tightening update coupling → the
  system is over-stepping. The persistent signature across all runs is policy
  thrashing (cold streaks) on a noisy ~18–35% band. Halving the LR, now safe under
  stable Huber loss, should let the policy settle into a steadier, higher floor.
- **Change:** `agent.py` Adam `lr=0.0001` → `lr=0.00005`. (Clean Huber baseline.)
- **Tag/branch:** `tuning-lr` (= LR 5e-5).
- **Run:** 228 — completed full 1000 episodes (32 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~34–37% (tail count) vs ~32% — modestly up.
  - Q-loss smoothed **2.61**, max spike **154** — best of every run so far.
  - Cold streaks fewer/shorter (longest dry patch ~8 vs 20–30 prior).
- **Verdict: WIN — KEEP.** Reward, Q-stability, and cold-streaks all improved
  *together* (coherent, not the noisy reward-only bump gamma gave). Folded into
  `tuning`. New baseline = Huber + LR 5e-5, ~35% success. Evidence: calmer = better.

### Exp 6 — learning rate 5e-5 → 3e-5 (bracket the LR optimum)
- **Hypothesis:** LR 5e-5 beat 1e-4 on every axis; continue the winning direction
  to find the optimum. 3e-5 may further steady the policy (fewer cold streaks,
  higher floor) — or undertrain within 1000 episodes, which would show as a still-
  rising curve / lower success → revert. Either way we bracket the optimum.
- **Change:** `agent.py` Adam `lr=0.00005` → `lr=0.00003`.
- **Tag/branch:** `tuning-lr3e5`.
- **Run:** 229 — completed full 1000 episodes (33 min).
- **Result vs baseline (Huber + LR 5e-5, ~35%):**
  - success rate ~24–27% (tail count) vs ~35%; peak episode_reward 0.54 vs 0.59.
  - Q-loss smoothed 3.45 vs 2.61; a ~20-episode cold streak returned (884–905).
  - Likely undertrained in 1000 episodes (curve still rising).
- **Verdict: REGRESSION → REVERT.** LR optimum is bracketed:
  **3e-5 (~25%) < 1e-4 (~32%) < 5e-5 (~35%).** Kept LR 5e-5.

### Exp 7 — batch size 64 → 128 (continue the "steadier updates" theme)
- **Hypothesis:** both wins (Huber, lower LR) came from steadier updates. The
  untried lever in that family is batch size: larger batches cut gradient variance
  directly, which should reduce cold streaks / raise the floor past ~35%.
- **Change:** `train.py` `agent.train(..., batch_size=64)` → `batch_size=128`.
- **Tag/branch:** `tuning-batch`.
- **Run:** 230 — completed full 1000 episodes (40 min, slower).
- **Result vs baseline (Huber + LR 5e-5, ~35%):**
  - success rate ~24–34% (tail count, avg ~30%) vs ~35%; peak 0.60 ≈ 0.59.
  - Q-loss smoothed 3.70 vs 2.61; cold streaks back (15–19 ep dry patches).
- **Verdict: NEUTRAL → REVERT.** No gain, slightly noisier, slower. Kept batch 64.

### Exp 8 — episodes 1000 → 2500 (undertraining vs structural ceiling)
- **Hypothesis:** the decisive test before declaring a ceiling. *Every* run's
  reward curve is still trending "improving" at ep 999, so ~35% may be
  undertraining, not a true ceiling. 1000 episodes is a pre-existing default (not
  a deliberate user choice). 2.5× the training on the best baseline either climbs
  well past 35% (→ undertrained, keep going) or plateaus (→ structural ceiling;
  remaining gap needs architecture/exploration/reward changes — a design decision
  for Robert, not single-variable tuning → end loop and report).
- **Change:** `train.py` `episodes=1000` → `episodes=2500`. (Best baseline:
  Huber + LR 5e-5 + batch 64 + K=4 + gamma 0.99.)
- **Tag/branch:** `tuning-episodes`.
- **Run:** 231 — completed full 2500 episodes (1h22m).
- **Result vs baseline (Huber + LR 5e-5, ~35%):**
  - success rate at ep 1000 ~35%, ep 1850 ~35%, ep 2400–2499 ~30–36%.
    **Flat across 2.5× the training.** Reward EMA peaked ~0.66 near ep 1870 then
    settled back to ~0.49; cold streaks persisted throughout (e.g. 2351–2366).
  - Q-loss drifted up late (smoothed ~11 by ep 2500) — more training did not even
    keep loss tighter.
- **Verdict: NO BENEFIT → REVERT** to episodes=1000. **The ~35% plateau is a
  structural ceiling, not undertraining.** This was the decisive test.

> **Meta-note:** n=1 run per config; the 18–35% reward band is partly seed noise.
> Huber is the one robust win (Q-loss 69→3 is consistent, not noise). LR 5e-5
> landed at the top of the band with the cleanest stability, so it was kept; the
> ceiling test (Exp 8) confirmed ~35% is a noise-limited plateau — the remaining
> gap is representational/exploration, not a single hyperparameter.

---

## Final Summary (session end)

**Result.** Took the agent from a diverging-Q baseline (~26%, Q-loss spiking to
4600) to a **stable ~35% success rate** on `collect_trash`. Two changes did all
the work, and both are about *calmer, steadier updates*:

| # | Change | Verdict | Why |
|---|--------|---------|-----|
| 1 | MSE → **Huber loss** | ✅ KEEP | Killed Q divergence (loss 69→3 smoothed). |
| 5 | LR 1e-4 → **5e-5** | ✅ KEEP | Best stability + top of reward band; fewer cold streaks. |
| 2 | gamma 0.99→0.995 | ↩ revert | Reward gain within noise, cost Q-stability. |
| 3 | HER K 4→8 | ↩ revert | Buffer flooded with easy near goals; regressed to ~21%. |
| 4 | Polyak τ=0.005 | ↩ revert | Tighter target coupling → *more* thrash (~18%). |
| 6 | LR 5e-5→3e-5 | ↩ revert | Undertrained/worse (~25%); brackets optimum at 5e-5. |
| 7 | batch 64→128 | ↩ revert | No gain, slightly noisier, slower. |
| 8 | episodes 1000→2500 | ↩ revert | **Flat at ~35% — ceiling, not undertraining.** |

**Current best config on `tuning`:** Huber loss · LR 5e-5 · batch 64 · HER K=4 ·
gamma 0.99 · 800 grad-steps/episode · epsilon 1.0→0.1 (min by ep 100) · hard
target update / 1000 steps · episodes 1000. TB runs 224 & 228 are the keepers.

**The wall (for discussion — structural, not tuning).** Failure mode is constant:
the agent solves quickly when it *starts near* the trash (1–120 steps) but burns
the full 1000 steps and scores 0 when it starts far. `best_score=1` every run, so
the architecture *can* represent the solution — it just can't reliably navigate
long distances. Single-hyperparameter tuning is exhausted; the remaining 35→~100%
gap needs a design change. Candidates, roughly in order of expected payoff:

1. **Start-distance curriculum** — begin episodes near the goal, expand the radius
   as success rises. The textbook fix for sparse long-horizon navigation; directly
   attacks the distant-start failure.
2. **Relative goal in the observation** — feed the goal as an egocentric vector
   (Δx, Δy) alongside the image, so the policy doesn't have to infer self-position
   from a 96×96 frame to plan a long path.
3. **Shorter `max_steps` during training** (e.g. 200–400) — denser learning signal
   per env-step and far faster iteration; lengthen later as a curriculum.
4. **n-step returns** — propagate sparse reward across long horizons more directly
   than a gamma bump (which we saw destabilizes).
5. **Exploration** — count/novelty bonus, or revisit the epsilon floor; the greedy
   policy may simply never *experience* reaching far goals to learn from.

Loop ended here per the stop condition: a structural ceiling, not a tuning knob.
All experiment branches cleaned up; `tuning` holds the best config.

---

## Day 2 (2026-06-10): The "structural ceiling" was a bug — ours, not the env's

### Exp 9 — gamma 0.995 redux (clean A/B on the best config)

- **Hypothesis:** the overnight gamma test (Exp 2) was confounded — it ran on
  the *old* LR 1e-4 baseline before the LR 5e-5 win landed. Re-test gamma 0.995
  on top of Huber + 5e-5.
- **Branch/tag:** `tuning-gamma-redux` · **Run:** 232
- **Result:** 100-ep success windows 21–27% all the way to the end (final two
  windows 27%, 26%) vs ~35% baseline. avg_q_loss spiked to 236–283 repeatedly —
  the hindsight Q-inflation bound is 1/(1-gamma), and 0.995 doubles it (see below).
- **Verdict:** ↩ REVERT, final. gamma 0.99 stands. The Day-1 verdict was right
  for a partially wrong reason.

### Code review: three structural bugs found (while Exp 9 ran)

1. **HER hindsight transitions never terminated.** The env terminates on success
   (`terminated = reward > 0.5`), so real successes store `done=True` → target
   exactly 1. Hindsight relabels reused the original `t.done` (≈always False), so
   relabeled successes bootstrapped past the goal: target 1 + γ·maxQ, compounding
   toward 1/(1-γ) ≈ 100. With K=4, ~80% of the buffer carried this corrupted
   signal. Huber loss had been masking the symptom since Exp 1.
2. **Truncation stored as terminal.** `done = term or trunc` went into the buffer,
   training Q toward exactly 0 at timeout states — i.e. far-from-goal states.
   This is precisely the observed failure mode (solves near, fails far).
3. **Unnormalized goal coords.** Raw map pixels (≤864) fed a Xavier-init Linear
   while the obs branch got /255 inputs. With n_trash=2 and identical sprites,
   the goal vector is the *only* disambiguator — and it was the worst-scaled input.

### Bugfix run — all three fixes

- **Branch/tag:** `bugfix-her-done` (commit 40cc148) · **Run:** 233 · gamma 0.99
- **Changes:** hindsight `done = reward > 0.5`; buffer stores `term` not
  `term or trunc`; `QModel.encode_goal()` scales goals to [0,1]. Regression test
  added (`test_hindsight_success_is_terminal`).
- **Result (100-ep success windows):** ramp to **56%** (eps 620–719), 49%
  (720–819), 47% (800–899), 35% (900–999, two cold streaks). Smoothed reward
  peaked **0.74** at ep 849 — baseline never broke ~0.55 in eight runs.
  avg_q_loss ~1e-4 smoothed, **zero** inflation spikes (Exp 9 spiked to 283).
  Far-start solves throughout: successes at 768, 851, 960, 638, 563 steps —
  the "can't navigate distance" failure mode is broken.
- **Verdict:** ✅ KEEP — biggest single win of the project. Day-1's "structural
  ceiling at 35%" was these bugs, primarily #1.
- **Caveat:** final-100 window dipped back to 35% (cold streaks eps 895–903,
  953–964). Late-run wobble is worth one confirming run and/or an epsilon-floor
  look before declaring a new plateau number.

### Where this leaves the knobs

The Day-1 hyperparameter verdicts were all measured against a corrupted target
landscape. Huber + LR 5e-5 likely remain sensible, but everything else deserves
a cheap re-check on fixed code if we keep tuning. Next levers, in order:
epsilon floor (0.1 → 0.05/decayed; 10% random actions is now the binding noise),
then the Day-1 structural list (curriculum, relative-goal obs) for the remaining
gap to consistent 1s.

### Exp 10 — HER K=2 + buffer 200k (episodes-of-history fix)

- **Hypothesis (Robert's):** run 233's late-run fade was catastrophic forgetting.
  At K=4/100k a failed episode writes ~6000 transitions, so the buffer held only
  ~16 episodes of history. K=2 + 200k → ~66 episodes retained, *and* a healthier
  1:2 real:hindsight ratio. (Logged as one experiment: the single conceptual
  variable is buffer history depth.)
- **Branch/tag:** `tuning-k2-buf200k` · **Run:** 234 · VRAM 11.0GB as predicted
- **Result (100-ep success windows):** 49% (720–819), 51% (820–919), **64%**
  (887–986). Smoothed reward 0.72 at run end, peak **0.86 at ep 983** — still
  climbing when the run hit the 1000-episode cap. Longest cold streak 8 episodes
  (233 had 12). The final window is the strongest stretch of the entire project;
  233's pattern (fade to 35%) is gone.
- **Verdict:** ✅ KEEP. Forgetting hypothesis confirmed.
- **Note:** Exp 8's "more episodes is flat" verdict is stale — it was measured on
  the bugged HER code. The curve now rises through ep 1000, so episodes=2500 is
  re-queued as Exp 11 (branch `tuning-episodes-2500`, prepared but not launched —
  training server reserved for the evening). Decision rationale: more episodes
  over more grad-steps/episode, because replay ratio is already ~8:1 at run end
  and the missing ingredient is fresh far-start data, not more passes over the
  buffer. Watch for a stall near 0.8 average — that may be the epsilon-0.1 floor,
  not a ceiling; check greedy test() before tuning further.
