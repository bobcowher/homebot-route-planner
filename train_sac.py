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
    # alpha_lr 1e-4 -> 3e-5: slow the temperature controller so it stops yanking the policy
    # out of commitment (run 346 alpha cycled 0.05<->0.20 over ~100 eps). A gentler controller
    # lets a committed, reaching policy persist and compound instead of oscillating.
    alpha_lr=3e-5,
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
# Start-distance curriculum (ADAPTIVE — the exploration fix): the discrete policy moves
# 4px/step, so a random walk only diffuses ~126px over a 1000-step episode — far spawns
# are physically unreachable and yield no learning signal (run 337/338: reward ~0, the
# agent only "won" when randomly spawned inside the goal). Spawn the robot CLOSE to the
# goal (120px) so navigation is short enough to actually reach, then expand the spawn
# distance toward the full map only once the agent clears a 60% reach-rate at the current
# distance (a fixed schedule outran learning in run 339). Uses the env's normal 79px reward.
agent.train(episodes=1200, batch_size=64, warmup_steps=5000,
            start_dist_start=120.0, start_dist_max=900.0, start_dist_step=15.0,
            start_dist_window=25, start_dist_threshold=0.6, start_dist_min=90.0)
