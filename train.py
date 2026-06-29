"""Train discrete SAC + HER on the HomeBot2D collect_trash leg (full random start).

The recipe (actor-driven discrete SAC):
  - 4x512 double-Q critic + categorical actor.
  - HER (relabel to achieved goals) — this is the curriculum; no spawn/reach curriculum.
  - HARD-VALUE critic: min-double-Q target with the entropy term DROPPED from the bootstrap
    (V=Σπ·minQ, no −α·logπ) + plain MSE + polyak tau=0.005, static alpha=0.1. The soft target's
    α·H/(1−γ) entropy offset floods mean_q to ~10 and buries the HER goal-advantage (run 389
    image-blind diag proved the flood is in the bootstrap, not the representation). Entropy is
    kept in the ACTOR loss so the policy stays stochastic. (avg/Q-clip/hard-sync patches reverted.)
  - N-STEP RETURNS (n_step=8): the funnel diagnostic (scripts/diagnose_qspread.py) on run 394
    (n=3) overturned the "V is flat" read — the critic DID learn a clean distance-to-goal value
    field, but only a ~150px FUNNEL around the goal; beyond ~150px V is flat ~0 out to map scale
    (864x576), so most random-start spawns get ZERO directional signal -> reach stalls ~15-20%.
    γ_eff was harsher than the true 0.99 with distance (far value UNDER-PROPAGATING): n=3 didn't
    flow the terminal reward back across the map. n=8 propagates it 8 steps/update to WIDEN the
    funnel (the in-funnel per-action advantage is already correct + the actor reads it; the wall
    is funnel WIDTH, not value-blindness). HER relabels get n-step returns too. Success metric =
    funnel width via the diagnostic on a checkpoint (seconds), not a 1500-ep entropy trend.
  - Behaviour = SAMPLE the stochastic actor (agent.sample_actor_action), with a DECAYING
    fraction of WHOLE episodes run as pure front-biased directed traversals (Q-schedule
    1.0 -> 0.25 floor). Sparse 0/1 reward gives no advantage on its own; the directed
    episodes feed HER clean map-crossing goal-reaching trajectories so the critic learns a
    real far-goal advantage for the actor to concentrate onto. Actor episodes read true reach.
  - max_steps=1000: long trajectories give HER rich relabel data and let the agent
    traverse to far random-start goals.
"""
import gymnasium as gym
import homebot  # noqa: F401  (registers HomeBot2D-Goal-V1)

from agent import SACAgent

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=1,
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,
)

agent = SACAgent(
    env=env,
    max_buffer_size=200000,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,    # Static 0.1 (Robert's best-success value; auto-tune never converged
                  # usefully here). alpha=0.01 was only ever needed to keep the critic's
                  # ARGMAX reachable for the OLD argmax-over-critic behaviour — obsolete now
                  # that behaviour SAMPLES the actor. At 0.01 the actor's entropy collapsed
                  # to ~0 by ep59 (run 377) -> deterministic actor -> the same wobble/
                  # oscillation SAC exists to avoid. With actor-driven behaviour the entropy
                  # is the FEATURE (exploration + anti-oscillation), so alpha goes back up.
    lr=3e-4,
    goal_noise_std=0.0,   # was 30.0 — an anti-vibration patch (break position->action
                          # memorization), now redundant: the sampled stochastic actor is the
                          # anti-oscillation mechanism. And 30px ≈ GOAL_RADIUS(31), so the noise
                          # SWAMPED the goal vector at reach scale — and HER relabels only the
                          # vector (image unchanged), so HER's close-range positive signal was
                          # taught through pure noise = unlearnable advantage (runs 380-385).
    head_layers=4,
    head_hidden=512,
    n_step=8,     # n-step return horizon. 3 (run 394) was too small to widen the ~150px value
                  # funnel the diagnostic measured; 8 propagates the terminal reward 8 steps/update
                  # back across the map to extend the funnel. Tradeoff: larger n = more off-policy
                  # bias (intermediate actions are the behaviour policy's, not the current actor's)
                  # — acceptable here, the directed-episode reaches are what we WANT propagated.
                  # READ via scripts/diagnose_qspread.py on a checkpoint: did V stay positive past
                  # ~150px (funnel widened)? If yes but partial, push n higher (12-16) or raise γ.
)

agent.train(episodes=2000, batch_size=64, warmup_steps=5000,
            explore_start=1.0, explore_min=0.25, explore_decay=0.977)
