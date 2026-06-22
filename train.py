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

# MACRO-ACTION HEAD (H=3) on champion 314. The 8-way / 4px action space can only head
# in 8 directions, so when the goal bearing falls between two compass actions the
# greedy policy dithers (N<->NE) -- the spinning we keep fighting. A length-3 macro
# (8**3 = 512-way head) lets the policy commit a SEQUENCE whose centroid expresses
# in-between headings ([N,NE,N] ~ NNE) the single-step space can't -- attacking the
# discretization root cause. (Action-repeat could only repeat ONE action; this TRAINS
# on the joint sequence -- the fundamental version.) Open-loop at train AND deploy +
# single-reward-at-landing SMDP backup (gamma**3); 8 base actions (STOP deferred -> no
# env change). Curriculum OFF (reach_start unset) to isolate the macro variable from
# run 322 (the reach-gradient experiment) -- this is a parallel A/B.
# Judge vs 314 (4.30/38% chain, 5.8% deploy spin) + run 322: chained_eval.py +
# spin_metric. Watch spin drop and collect_trash reach climb -- and WATCH
# overestimation (max over 512 actions amplifies positive bias; Double-DQN is on).
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, motion_window=1, random_goal_tiles=True, macro_h=3)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600)
