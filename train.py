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
# V2 STABILIZATION (run 323 v1 collapsed): v1 learned (greedy reach peaked 0.70 ~ep
# 550) then diverged via VALUE OVERESTIMATION -- avg_q_loss blew up to ~6.9, reach
# crashed to 0.05, chain 1.9->0.2. The max over 512 macros amplifies positive bias,
# and the collapse tracked the HER anneal (her_anneal_start=600) stripping the dense
# relabel grounding. Two targeted stabilizers, both pointing the same way:
#   head_norm=True   -- LayerNorm in the head, there to curb overestimation
#   her_anneal_start=None -- keep HER's grounding for the whole run (512 actions can't
#                            afford to lose it). Keep H=3 (it showed life at H=3); if
#                            this still collapses, drop to H=2 (64 actions) next.
# Judge vs 314 (4.30/38% chain, 5.8% deploy spin): chained_eval.py + spin_metric.
# WATCH avg_q_loss -- if it climbs again, overestimation isn't beaten and H=2 is next.
# NOTE: head_norm=True -> post-hoc eval (load_q_model) must pass head_norm=True.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              head_norm=True, use_motion=True, motion_window=1,
              random_goal_tiles=True, macro_h=3)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=None)
