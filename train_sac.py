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
    # alpha (temperature) is now DECAYED on a fixed schedule (epsilon-greedy analog),
    # not auto-tuned. Run 351 showed the auto-tuner's max-entropy actor never exploits the
    # HER-built critic (entropy pinned at max for 1100 eps, 0 reaches). We disable autotune
    # and decay alpha 0.2 -> 0.01 (explore -> argmax-Q exploit) in agent.train() below.
    alpha=0.2,                  # starting temperature for the decay
    lr=3e-4,
    goal_noise_std=30.0,
    autotune_alpha=False,
    # Asymmetric heads (Robert's hypothesis): actor DEEP (4x512 — compositional policy,
    # like the Q-champion's head_layers=4), critic FLAT & WIDE (2x1024 — value regression
    # where depth amplifies bootstrap overestimation; wide-shallow is steadier and should
    # give cleaner Q-spread for the actor to exploit, vs run 352's early entropy collapse).
    actor_head_layers=4, actor_head_hidden=512,
    critic_head_layers=2, critic_head_hidden=1024,
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
# Alpha decay 0.2 -> 0.01 over the first 60% of episodes, then hold (epsilon-greedy analog):
# high temperature early so the policy explores and feeds HER diverse trajectories, decaying
# to near-argmax late so the actor exploits the HER critic and closes the bootstrap loop that
# DQN gets for free (behaviour = argmax-Q). 0.01 keeps a little residual stochasticity (~ the
# champion's epsilon=0.1 floor).
agent.train(episodes=1200, batch_size=64, warmup_steps=5000,
            alpha_anneal_to=0.01, alpha_anneal_episodes=720)
