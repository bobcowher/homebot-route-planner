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

# Depth ladder, rung 4: 4-layer head (all 512), constant width. Depth paid
# monotonically 1->2->3 (greedy/softmax peak AND EMA); does it still climb at 4,
# or plateau / destabilize (plain MLP, no residuals)? goal encoder held at 2.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4)

agent.train(episodes=1200, batch_size=64, eval_interval=50, eval_episodes=20)
