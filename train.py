from agent import Agent
import gymnasium as gym
import homebot

# REAL-REWARD REACH FIX (champion 314 architecture). The whack-a-mole "reach"
# failure traced to a reward bug: the goal env rewarded + terminated every goal at a
# flat GOAL_THRESHOLD=79px (compute_reward), but the task actually collects trash only
# at 31px (tasks.py). We trained the policy to park at 79 and graded it at 31.
# FIX (no per-target hand-engineering):
#   - env.step now uses the TaskManager's TRUE per-target reward and terminates on
#     real task completion (is_done), not a geometric radius.
#   - compute_reward is demoted to the HER hindsight relabel proxy ONLY, at a single
#     tight RELABEL_RADIUS=31 (the tightest real radius; satisfies door/fixtures too).
# Goal = the trash pile itself (random_goal_tiles OFF). n_trash=1 so the single trash
# == the conditioned goal == the completion event (at n_trash=2 the real reward fires
# for EITHER pile, decoupling reward from the goal coord). spawn_trash is uniform over
# valid floor tiles, so a single random trash keeps full-map coverage -- we get an
# arbitrary-coord reacher AND a real reward, not a fixed-spot specialist.
# Clean A/B vs 314 (4.30/38% chain, 5.8% deploy spin) on the collect_trash leg:
# chained_eval.py + spin_metric. her_anneal_start=None keeps HER's dense relabel
# grounding the whole run (the tight 31px target needs it).
#
# ROLLOUT POLICY: pure softmax @ temp 0.01, NOT softmax_rel or argmax. The Q
# values ARE the probability distribution: P(a) = exp(q_a/tau) / Z. Exploration
# is value-driven (a 0.2-Q action is picked 20% of the time, period) and
# auto-anneals -- early flat Q -> near-uniform (explore), learned Q -> peaked
# (exploit). Distance-aware for free: far states with flat Q stay explorative
# (breaking transit limit cycles), near-goal states with peaked Q go greedy.
# tau=0.01 is the absolute temp that scored 0.90 reach_rate on run-325's eval
# (vs 0.45 at 0.025, 0.25 at 0.05). epsilon is kept for early uniform coverage
# but min_epsilon=0 so it anneals to ~0 by ~ep300; the last ~1500 eps are pure
# softmax. A/B vs the softmax-behavior branch (softmax_rel @ temp 0.1).
env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=1,           # single trash == conditioned goal == completion event
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,   # env owns spawn (uniform valid tile, >=60px from goals)
)

agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, motion_window=1,
              softmax_behavior=True, softmax_behavior_temp=0.01,
              min_epsilon=0.0)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=None)
