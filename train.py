from agent import Agent
import gymnasium as gym
import homebot

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=2,
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,   # env owns spawn now (uniform valid tile, >=60px from goals)
)

# Coords + deep head + information bottleneck. Coords-deep overfit (train 0.94 >>
# greedy eval 0.70); squeeze the fused decision rep through a 64-wide channel
# before the action head to force compression and close the memorization gap.
agent = Agent(env=env, max_buffer_size=200000,
              goal_layers=2, head_layers=2, bottleneck=64)

agent.train(episodes=1200, batch_size=64, eval_interval=50, eval_episodes=20)
