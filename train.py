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

# Random-goal training on the locked depth-4 navigator (goal_layers=2/head=4).
# random_goal_tiles=True: each episode's goal is a uniformly-sampled valid floor
# tile (whole-map coverage) instead of a trash spot. chained_eval showed the
# trash-only navigator generalizes to interior fixtures (45-65%) but fails the
# east doorway (~0-15%) — a coord trash never spawns near. Whole-map goal
# sampling fills that coverage gap. Train+eval both use random tiles.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              random_goal_tiles=True)

agent.train(episodes=1200, batch_size=64, eval_interval=50, eval_episodes=20)
