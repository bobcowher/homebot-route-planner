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

# SUCCESS-RADIUS CURRICULUM on champion 314. ROOT CAUSE: the env rewards AND
# terminates at GOAL_THRESHOLD=79px, but collect_trash is evaluated at 31px. So the
# navigator never experiences the 31-79px shell during training -- the value is flat
# there, no gradient to close the last 48px -> it stalls/vibrates far from goal
# ("never gets close, trash not in view"). collect_trash reaches only 52% (vs 99%
# for fixtures, which are graded at a looser 79px). Fix: anneal the reach+terminal
# radius 79 -> 28 over eps 100-600 (while HER hindsight is full, so the tighter bar
# always has dense relabel signal). Pure HER artifact -- no env change (the env's
# fixed 79px reward/termination is ignored; reward/term recomputed from robot pose).
# Config = champion 314: depth-4, velocity-only motion (motion_window=1), epsilon-
# greedy + hard-Q. NOTE: episodes now run until within the (shrinking) radius, so
# late episodes are longer -> slower than the 35-min softmax runs.
# Judge: chained_eval.py --readouts greedy softmax_rel --temp 0.1 + spin_metric, vs
# 314 best (4.30/38% chain, 5.8% deploy spin). Watch collect_trash reach rate climb.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, motion_window=1, random_goal_tiles=True)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600,
            reach_start=79.0, reach_end=28.0,
            reach_anneal_start=100, reach_anneal_end=600)
