"""Train discrete SAC + HER on the HomeBot2D collect_trash leg (full random start).

The recipe (actor-driven discrete SAC):
  - 4x512 double-Q critic + categorical actor.
  - HER (relabel to achieved goals) — this is the curriculum; no spawn/reach curriculum.
  - HARD-VALUE critic: min-double-Q target with the entropy term DROPPED from the bootstrap
    (V=Σπ·minQ, no −α·logπ) + plain MSE + polyak tau=0.005, static alpha=0.1. The soft target's
    α·H/(1−γ) entropy offset floods mean_q to ~10 and buries the HER goal-advantage (run 389
    image-blind diag proved the flood is in the bootstrap, not the representation). Entropy is
    kept in the ACTOR loss so the policy stays stochastic. (avg/Q-clip/hard-sync patches reverted.)
  - N-STEP RETURNS (n_step=3): the hard-value bootstrap (run 392) gave the first sustained
    actor reaches (~15-20%) but then PLATEAUED — run 393 (alpha 0.05) showed entropy pinned at
    max over 1100 eps because the per-action Q-spread Δ≈0: under sparse 0/1 + γ=0.99 + 1-step
    bootstrap, Q(s,a)=r+γV(s') has no spatial gradient for the actor to concentrate onto.
    3-step returns propagate the terminal reward 3 steps back (recomputed per-step inside HER
    too), carving the toward-vs-away gradient. No shaping/env-change; ports to continuous.
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
    n_step=3,     # n-step return horizon. 3 is the conservative first probe (lower off-policy
                  # bias than 5; the directed-episode data is heavily off-policy early). If the
                  # actor entropy drops + reach climbs past the run-392 ~20% plateau -> n-step is
                  # the right lever; bump toward 5 for a deeper gradient. Flat -> try a dueling
                  # /advantage head instead (the alt on the credit-assignment axis).
)

agent.train(episodes=2000, batch_size=64, warmup_steps=5000,
            explore_start=1.0, explore_min=0.25, explore_decay=0.977)
