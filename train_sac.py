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
# Success-radius curriculum: bounded-alpha SAC stays stable but doesn't *learn* —
# random walks almost never hit the 79px goal, so the critic gets no real reward and
# the policy never commits (run 337: reward stuck at 0, entropy ~1.9). Start with a
# big 200px reach radius so real goal-reward floods the critic early, then anneal to
# the env's 79px over the first 600 episodes (hold 79 after). Reaching also terminates
# episodes, which further tames the soft-value entropy bonus.
agent.train(episodes=900, batch_size=64, warmup_steps=5000,
            reach_start=200.0, reach_end=79.0,
            reach_anneal_start=0, reach_anneal_end=600)
