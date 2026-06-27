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
    # max_steps 1000 (champion fidelity). 250 was a run-341 fix to bound the Q-divergence
    # horizon — but that divergence belonged to the undersized 2x256 critic + the autotuner's
    # entropy-bonus runaway, BOTH now gone (4x512 critic, fixed alpha 0.1, critic-greedy
    # behaviour). The champion used 1000, and on random_start it matters: longer trajectories
    # give HER far more/better relabel data and let the agent actually traverse to far goals.
    # Every SAC run plateaued at 3-10% with 250; this restores the champion's data regime.
    max_steps=1000,
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
    # FIXED alpha 0.1, autotune OFF. Auto-entropy-tuning never converged usefully here (every
    # variant oscillated, collapsed, or went inert — runs 336/342/347/348/354) — matches
    # Robert's experience that SAC auto-alpha reliably lands on static ~0.1 anyway. Exploration
    # is moved to epsilon-greedy ARGMAX behaviour (below), so alpha is just a mild actor entropy
    # regulariser, not the explore/exploit knob.
    alpha=0.1,
    lr=3e-4,
    goal_noise_std=30.0,
    autotune_alpha=False,
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
# Epsilon-greedy ARGMAX behaviour (the champion's exploration, faithfully): random action
# w.p. epsilon, else argmax(policy). epsilon 1.0 -> 0.1, decay 0.977/episode (champion values;
# hits 0.1 by ~ep100). Argmax COMMITS regardless of Q-spread size — a low-temp softmax does
# not (runs 352/354 only committed when a lucky Q-spike happened to force it). This is the
# mechanism that lets the actor exploit the HER critic and close DQN's virtuous loop.
agent.train(episodes=1200, batch_size=64, warmup_steps=5000,
            epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.977)
