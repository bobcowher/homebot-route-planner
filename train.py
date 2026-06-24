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
# GOAL REPRESENTATION: noisy world vector [dx, dy] + N(0, 30px²) instead of
# absolute coords [robot_x, robot_y, goal_x, goal_y]. Relative displacement so
# the network can't memorize a position→action lookup (the memorization path
# that the coord rep left open). Gaussian noise (30px ≈ 1 tile ≈ ~47cm) simulates
# the shaky-map reality and breaks the memorization key further — same position
# gives a different noisy vector each visit, forcing a smooth "approach the goal"
# skill rather than a lookup table. Hypothesis: this is the rep change that makes
# the value field less flat far from goal (the root cause of transit cycles).
# A/B vs 325 (absolute coords, no noise) on chained_eval + spin_metric + the new
# maps Robert is building (generalization test).
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
              goal_noise_std=30.0)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=None)
