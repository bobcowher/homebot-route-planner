from agent import Agent
import gymnasium as gym
import homebot

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="human",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=2,
    max_steps=1000,
    map_name="default",
    goals=["go_to_fridge", "deliver_drink", "go_to_door", "deliver_package", "collect_trash"],
    evaluate=True,
)

agent = Agent(env=env)

agent.load()

agent.test(episodes=10)
