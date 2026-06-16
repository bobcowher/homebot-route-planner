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

# depth-4 FUNNEL: 3x 512 head + a 128 taper before the output (512->512->512->128
# ->out). The 4th layer adds depth AND a mild compression, vs depth-4's 4th full
# 512. A/B against depth-4 (run 291): does a tapered 4th layer match/beat a full
# one, distilling the decision rep more cheaply? goal encoder held at 2.
agent = Agent(env=env, max_buffer_size=200000,
              goal_layers=2, head_layers=3, bottleneck=128)

agent.train(episodes=1200, batch_size=64, eval_interval=50, eval_episodes=20)
