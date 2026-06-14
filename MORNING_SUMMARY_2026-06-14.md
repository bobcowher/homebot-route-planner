# Overnight Results — bearing-conditioned DQN+HER reacher (run 260)

**TL;DR: It failed the same way the SAC reacher did.** The env is solvable and
exploration reaches the goal repeatedly, but the *greedy* policy never learns
to. No code bug — this is a design problem, and I think I know which one.

---

## What ran

- Branch: `bearing-reacher` (off `her`, reusing QModel/buffer/HER/Double-DQN).
- Goal = egocentric bearing only `[sin(b), cos(b)]`, no range. Random spawn each
  episode. Un-gameable budget-limited greedy eval.
- 29/29 tests passed. Started clean on Beekeeper (no `_build_obs` crash).
- Run 260, RTX 3090, ~4s/episode.

## The numbers

`Eval/greedy_reach_rate` across the whole exploitation phase (epsilon floored at
0.1 since ep98):

| episode | 0 | 50 | 100 | 150 | 200 | 250 |
|--------:|---|----|-----|-----|-----|-----|
| reach   |0.05|0.00|0.00|0.00|0.00|0.05|

Flat at zero. The two 0.05s are a single lucky episode each (1/20).

Meanwhile **exploration** (epsilon=0.1, 90% greedy + 10% noise) reaches the goal
constantly: ep159 (286 steps), 160 (120), 164 (169), 179 (186), 204 (177), etc.

`Train/avg_q_loss` collapsed to ~2e-5 — a self-consistent fixed point.

## The tell

90%-greedy reaches the goal; **100%-greedy reaches it 0/20**. The only difference
is the 10% random kick. So pure-greedy gets stuck in a limit cycle (orbits the
goal / faces a wall / oscillates) and the noise is the only thing breaking it
out. The value function converged to something self-consistent but not
task-useful — *the same failure family as the SAC reacher.*

## What I ruled out (read the code, didn't just watch logs)

- **Train/eval mismatch:** none. Both paths compute bearing identically and
  normalize obs `/255.0` identically (`agent.py` select_action vs greedy_eval).
- **HER pipeline:** clean. `episode_buffer.send_to` relabels bearings
  consistently and even handles `hindsight_done = reward > 0.5` to prevent the
  Q-inflation we hit before. Not the bug.

## My diagnosis (for discussion, not acted on)

Two things, and I think #2 is the real one:

1. **Goal dilution.** The bearing enters QModel as 128 dims concatenated with
   **4096** conv features (~3% of the fused vector). This is exactly what the SAC
   probe showed — the goal moved the action ~5%. The 2-dim goal is structurally
   drowned.

2. **The observation is a top-down god's-eye RGB render.** Two problems:
   - It's the thing drowning the goal (4096 features of map view vs 2 of bearing).
   - **It violates the core env-realism rule.** A real home robot has no
     overhead map view of itself. A bearing-driven reacher should see something
     *egocentric and small* — forward depth / a handful of obstacle ray-casts —
     not a top-down picture of the whole room. With random spawn + random trash
     position, the top-down image carries no consistent goal signal anyway, so
     the conv tower trains on noise and the policy collapses to direction-blind.

The bearing-only choice itself is probably fine — you don't need range to *drive
toward* something; the env auto-terminates on reach, so "turn until cos≈1, go
forward" is a complete policy. Range only matters for knowing when to stop, which
the env handles. So I don't think dropping range was the mistake.

## Proposed fix (your call in the morning)

Replace the top-down RGB observation for the C-tier reacher with an **egocentric,
low-dim observation**: bearing + N forward/side obstacle ray-casts (distances).
This:
- makes the goal signal dominant instead of drowned,
- is dramatically cheaper to learn (small MLP, no conv tower),
- and is *more* realistic (matches what a real robot's depth sensor gives),
  satisfying the core rule instead of violating it.

This is a meaningful redesign of the obs space, so I left it for you rather than
thrashing it overnight. Open questions for us:
- ray-cast count / FOV / max range?
- keep bearing coordinate-free, or allow a coarse "near/far" bucket (defensible:
  a real depth sensor gives rough range to a *detected* object)?

## Run status

Run 260 left running (harmless, nothing else queued — it's live evidence in TB).
Say the word and I'll stop it.
