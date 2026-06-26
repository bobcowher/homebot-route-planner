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
    alpha=0.1,                  # initial temperature; auto-tuned from here
    lr=3e-4,
    goal_noise_std=30.0,
    autotune_alpha=True,
    target_entropy_ratio=0.7,   # target entropy = 0.7 * log(8) ≈ 1.45 nats
)

# Warmup: fill the buffer with random transitions before any gradient update,
# so the critic doesn't bootstrap off a near-empty, undiverse buffer and blow up
# (the run-334 mean_q -> 104 divergence). 5k random steps.
#
# Start-distance curriculum (the exploration fix): the discrete policy moves 4px/step,
# so a random walk only diffuses ~126px over a 1000-step episode — far spawns are
# physically unreachable and yield no learning signal (run 337/338: reward ~0, the
# agent only "won" when randomly spawned inside the goal). Instead spawn the robot
# CLOSE to the goal (120-150px) so navigation is short enough to actually reach — real
# reward + HER on directed trajectories — then expand the spawn distance toward the
# full map (~900px) over 700 episodes. Uses the env's normal 79px reach reward.
agent.train(episodes=900, batch_size=64, warmup_steps=5000,
            start_dist_start=150.0, start_dist_end=900.0,
            start_dist_anneal_start=0, start_dist_anneal_end=700,
            start_dist_min=90.0)
