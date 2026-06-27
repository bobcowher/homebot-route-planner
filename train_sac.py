"""Discrete SAC + HER on the collect_trash leg.

CNN-based policy: image (96×96 RGB) + noisy_world_vector goal + motion.
Discrete action space (8 actions) — same as the DQN champion.
"""
import gymnasium as gym
import homebot  # noqa: F401

from sac_agent import SACAgent

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=1,
    # 250 steps * 4px/step = 1000px ~ the map diagonal, so a DIRECTED agent can still
    # cross the whole map, but a FAILING episode now bootstraps over 250 non-terminal
    # steps instead of 1000. That shrinks the soft-value horizon Sum(gamma^t) from ~100
    # to ~8, which is what re-inflated run-340's mean_q to ~60 (critic_loss 3021) and
    # collapsed the policy to a wrong deterministic behavior. Short curriculum distances
    # need far fewer than 250 steps anyway.
    max_steps=250,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,
)

agent = SACAgent(
    env=env,
    max_buffer_size=200000,
    # gamma 0.99: run 344 showed 0.95 over-contracts — the reward barely propagates
    # (0.95^130~0.001) so the critic learns Q~0 everywhere and reaches collapsed to ~1%.
    # Critic stability is now handled by CAPACITY (4x512 critic head, sac_model.py) — run
    # 346 showed the champion-sized critic recovers from commitment-induced Q-spikes that
    # diverged the undersized 2x256 — so gamma stays high for long-range credit assignment.
    gamma=0.99,
    tau=0.005,
    # Autotune ON, with the entropy TARGET annealed in agent.train() (controlled explore->
    # exploit). The tuner HOLDS entropy at the target (run 351 proved it), so a high target
    # early forces exploration and a low target late releases argmax-Q exploitation — fixing
    # run 352's premature collapse (fixed alpha decay let the policy commit at ep38).
    alpha=0.2,                  # initial temperature; tuner takes over
    lr=3e-4,
    goal_noise_std=30.0,
    autotune_alpha=True,
    alpha_lr=1e-4,
    alpha_min=0.005,            # low floor so alpha can reach near-argmax when target -> ~0
    alpha_max=1.0,
    # Symmetric deep 4x512 heads (run 353's flat-wide 2x1024 critic diverged 100x worse,
    # critic_loss -> 2e6, ~4% vs 352's ~10%: the value field needs depth, per the champion).
    actor_head_layers=4, actor_head_hidden=512,
    critic_head_layers=4, critic_head_hidden=512,
)

# Warmup: fill the buffer with random transitions before any gradient update,
# so the critic doesn't bootstrap off a near-empty, undiverse buffer and blow up
# (the run-334 mean_q -> 104 divergence). 5k random steps.
#
# NO start-distance curriculum. HER IS the curriculum: relabeling to achieved goals trains
# the agent on goals it actually reached (automatically easy-first, shrinking as it improves),
# so it learns goal-conditioned navigation from local moves without ever reaching the true
# goal — exactly how the DQN champion learned far-spawn reaching with random_start + HER and
# no start curriculum. The earlier diffusion argument for a curriculum was wrong for HER, and
# every no-curriculum SAC run (334/337) was under-capacity (2x256). This is the clean test:
# the champion recipe with the one proven SAC change (4x512 critic) + HER + env random_start.
# Entropy-target anneal 0.9*log(8) -> 0.02*log(8) over the first 70% of episodes, then hold.
# The auto-tuner holds entropy at the target, so this is a CONTROLLED explore->exploit: near-
# max entropy early (forces exploration, feeds HER diverse trajectories) -> ~0 late (releases
# the actor into argmax-Q to exploit the HER critic). Fixes run 352's ep38 premature collapse.
agent.train(episodes=1200, batch_size=64, warmup_steps=5000,
            target_entropy_ratio_start=0.9, target_entropy_ratio_end=0.02,
            te_anneal_episodes=840)
