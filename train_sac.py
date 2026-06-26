"""Discrete SAC + HER on the collect_trash leg.

CNN-based policy: image (96×96 RGB) + noisy_world_vector goal + motion.
Discrete action space (8 actions) — same as the DQN champion.
"""
import gymnasium as gym
import homebot  # noqa: F401

from sac_agent import SACAgent

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=1,
    # 250 steps * 4px/step = 1000px ~ the map diagonal, so a DIRECTED agent can still
    # cross the whole map, but a FAILING episode now bootstraps over 250 non-terminal
    # steps instead of 1000. That shrinks the soft-value horizon Sum(gamma^t) from ~100
    # to ~8, which is what re-inflated run-340's mean_q to ~60 (critic_loss 3021) and
    # collapsed the policy to a wrong deterministic behavior. Short curriculum distances
    # need far fewer than 250 steps anyway.
    max_steps=250,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,
)

agent = SACAgent(
    env=env,
    max_buffer_size=200000,
    # gamma 0.99: run 344 showed 0.95 over-contracts — the reward barely propagates
    # (0.95^130~0.001) so the critic learns Q~0 everywhere and reaches collapsed to ~1%.
    # Critic stability is now handled by CAPACITY (4x512 critic head, sac_model.py) — run
    # 346 showed the champion-sized critic recovers from commitment-induced Q-spikes that
    # diverged the undersized 2x256 — so gamma stays high for long-range credit assignment.
    gamma=0.99,
    tau=0.005,
    alpha=0.1,                  # initial temperature; auto-tuned from here
    lr=3e-4,
    goal_noise_std=30.0,
    autotune_alpha=True,
    # target 0.4 (0.83 nats), reverted from a 0.2 experiment: run 347 (target 0.2) showed
    # lowering the setpoint KILLS exploration — entropy stuck at 2.0, reaches ~1% (vs 346's
    # 14%). 346's burst-then-oscillate was the BEST behaviour; the setpoint was fine. The
    # oscillation is a DAMPING problem: when the policy commits (entropy < target) the
    # controller raises alpha and de-commits it. Fix is a SLOWER controller (alpha_lr below),
    # not a lower target — so commitment persists long enough to sustain >60% reach-rate.
    target_entropy_ratio=0.4,
    # alpha_lr 1e-4 (restored): run 348 (3e-5) showed the slow controller is HARMFUL — it
    # let the policy collapse to entropy 0 / a single fixed ~17-step path that only reaches
    # the ~5% of configs that suit it. 346's faster controller and its alpha oscillation were
    # PROTECTIVE: re-injecting exploration prevented that deterministic collapse (held 14%).
    alpha_lr=1e-4,
    # alpha_max raised 0.3 -> 1.0. The 0.3 ceiling was added in run 336 to cap the
    # entropy-bonus runaway when max_steps=1000 (Sum gamma^t ~ 100). With max_steps=250
    # that bonus is ~12x smaller, so a high alpha is safe — and run 342 showed the 0.3
    # ceiling is actively harmful: once the policy LEARNED to reach (real Q-spread), the
    # controller needed alpha > 0.3 to hold entropy at target, the ceiling blocked it,
    # entropy collapsed to one-hot and the soft value diverged (critic_loss -> 2.7e6).
    # A 1.0 ceiling lets the controller hold the 0.83-entropy operating point that was
    # learning well (24-step reaches) without collapse.
    alpha_max=1.0,
)

# Warmup: fill the buffer with random transitions before any gradient update,
# so the critic doesn't bootstrap off a near-empty, undiverse buffer and blow up
# (the run-334 mean_q -> 104 divergence). 5k random steps.
#
# NO start-distance curriculum. HER IS the curriculum: relabeling to achieved goals trains
# the agent on goals it actually reached (automatically easy-first, shrinking as it improves),
# so it learns goal-conditioned navigation from local moves without ever reaching the true
# goal — exactly how the DQN champion learned far-spawn reaching with random_start + HER and
# no start curriculum. The earlier diffusion argument for a curriculum was wrong for HER, and
# every no-curriculum SAC run (334/337) was under-capacity (2x256). This is the clean test:
# the champion recipe with the one proven SAC change (4x512 critic) + HER + env random_start.
agent.train(episodes=1200, batch_size=64, warmup_steps=5000)
