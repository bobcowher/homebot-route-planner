"""Train discrete SAC + HER on the HomeBot2D collect_trash leg (full random start).

The working recipe (learns full-map goal-conditioned navigation):
  - 4x512 double-Q critic + categorical actor.
  - HER (relabel to achieved goals) — this is the curriculum; no spawn/reach curriculum.
  - Fixed temperature alpha=0.1 (auto-tuning never converged usefully here).
  - Behaviour = epsilon-greedy ARGMAX over the critic (see agent.greedy_critic_action),
    epsilon 1.0 -> 0.1. The actor only feeds the soft-value bootstrap.
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
    alpha=0.1,
    lr=3e-4,
    goal_noise_std=30.0,
    head_layers=4,
    head_hidden=512,
)

agent.train(episodes=1200, batch_size=64, warmup_steps=5000,
            epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.977)
