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
    gamma=0.99,
    tau=0.005,
    alpha=0.1,                  # initial temperature; auto-tuned from here
    lr=3e-4,
    goal_noise_std=30.0,
    autotune_alpha=True,
    # 0.4*log(8) ≈ 0.83 nats. Lowered from 0.7 (≈1.45): with Q now bounded (max_steps
    # fix), run 341 sat at entropy 1.38 — ~4 effective actions, too stochastic to commit
    # to a directed ~30-step path, so reach-rate stalled at ~5%. A more committed policy
    # can execute directed navigation; the 0.05 alpha floor still prevents collapse, and
    # bounded Q removes the run-334 wrong-collapse risk that high entropy guarded against.
    target_entropy_ratio=0.4,
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
