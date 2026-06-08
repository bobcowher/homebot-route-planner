from agent import Agent
import gymnasium as gym
import homebot

env = gym.make(
    "HomeBot2D-v1",
    render_mode="rgb_array",   # "human", "rgb_array", or None
    action_mode="continuous",    # "discrete" or "continuous"
    obs_resolution=(96, 96),   # observation image size (H, W)
    n_trash=2,                 # trash items per episode
    max_steps=1000,            # steps before truncation
    map_name="default",             # map layout
    subgoals=True,            # True for LLM orchestration mode
)

agent = Agent(env=env, max_buffer_size=200000)

agent.train(episodes=1200, batch_size=32)
