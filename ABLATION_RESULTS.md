# Reacher Ablation Ladder — Results (2026-06-14)

## Headline

**Two findings:**
1. **The eval was the primary confound.** run 260's "0% greedy" (and the whole
   morning "bearing-reacher is broken / top-down RGB drowns the goal" story) was
   largely a **measurement artifact**. The over-strict greedy eval (tight 3×
   step budget + `distance<31` to the *specific* desired_goal) reported ~0% even
   for the **proven `her` config**. Fixing the eval to mirror `her/evaluate.py`
   (full episode up to max_steps, success = cumulative `reward>0.5`) recovered a
   real, non-zero, *efficient* greedy reach-rate.
2. **Stripping range (bearing-only) is the one realistic change that actually
   hurts.** Random spawn and the egocentric frame cost ~nothing; going to a
   unit bearing (direction without range) collapses reach-rate ~5×. This matches
   Robert's instinct that the old coordinate/displacement goal was load-bearing.

The earlier "top-down RGB observation drowns the goal / violates realism"
diagnosis is **RETRACTED** — the same conv+goal architecture works fine; the
image was never the problem.

## Method

One-variable-at-a-time ladder, each branch built off the proven `her` branch,
all measured with the **same fixed eval** (full-episode, `reward>0.5`, 20 eps,
+ `avg_success_steps` as a circling diagnostic). Env: `HomeBot2D-Goal`,
discrete, n_trash=2, max_steps=1000, goals=["collect_trash"]. Comparison at a
**matched ~ep150 budget** (see caveats).

| Rung | Branch | Variable added (vs previous) | greedy ep50 | greedy ep100 | greedy ep150 | avg_success_steps | run |
|---|---|---|---|---|---|---|---|
| 0 | abl0-her-honest-eval | `her` + fixed eval (baseline) | 0.00 | **0.25** | 0.20 | 17–43 | 263 |
| 1 | abl1-random-spawn | + random spawn each episode | 0.25 | — | 0.20 | 22 | 264 |
| 2 | abl2-egocentric | + egocentric goal frame (rotate by −heading, **range kept**) | 0.20 | — | 0.15 | 17–21 | 265 |
| 3 | abl3-bearing | + **strip range** → unit bearing | 0.00 | 0.05 | 0.05 | 1–32 | 266 |

(Rung 3 ep150=0.05 and its single "success" was a spawn-on-goal freebie
(avg_success_steps=1); ε-exploration also reaches the goal far less often than
the other rungs. Decisive.)

## Per-variable verdict

- **Random spawn (Rung 1): no harm.** ~0.20–0.25, equal to baseline. A real
  robot starting anywhere is fine.
- **Egocentric goal frame (Rung 2): ~no harm.** ~0.15–0.20, within 20-episode
  eval noise (~±0.1) of baseline. Rotating the goal into the robot frame is
  cheap — good, since it's the more realistic representation.
- **Range strip / bearing-only (Rung 3): REAL COST.** ~0.05 vs ~0.25 baseline
  (~5×), and exploration itself reaches the goal much less — direction-without-
  distance is a genuinely harder learning signal here. **Range matters.**

## Caveats

- **ep150 is a learning-speed snapshot, not the asymptote.** `her` reached ~43%
  greedy only after ~2500 episodes. These numbers are lower bounds; a slower-but-
  equal variant could look worse here than it really is. The Rung 3 gap is large
  enough to trust, but the absolute 0.25 is not the ceiling.
- 20-episode evals → ~±0.1 noise. Treat Rungs 0/1/2 as a tie.
- n_trash=2, so eval success = reaching *either* trash (matches `her`'s 43% eval).

## Long-run update (run 267, abl2-egocentric = recommended config)

Relaunched the recommended config (egocentric + random spawn + range kept) for a
long run to check the asymptote. Greedy reach-rate is climbing slowly/noisily:
ep400=0.20, ep450=0.15, **ep950=0.25** (avg_success_steps=15, efficient). Not yet
at `her`'s 0.43 but trending up — consistent with "needs more episodes" (`her`
took ~2500ep) rather than egocentric asymptoting lower. Still running.

## Recommended next steps

1. **Pick the config: `her` recipe + egocentric frame + random spawn + KEEP
   range** (Rung 2's design but with range, i.e. egocentric displacement). Run it
   long (~1500–2500 ep) to confirm it reaches the ~43% `her` asymptote under the
   honest eval.
2. **Decide the range policy (the realism conversation).** Range is necessary,
   but exact through-wall range is the oracle we wanted to avoid. Honest options:
   (a) range only when the goal is in line-of-sight (depth sensor), (b) a coarse
   near/far bucket the B-tier grounding can supply for a detected object. Test
   whether coarse range recovers most of the Rung-3 loss.
3. Keep the fixed eval. Optionally add back a *generous* budget (e.g. 6–8×) as a
   secondary efficiency metric — but never as the primary gate.
