"""Train discrete SAC + HER on the HomeBot2D collect_trash leg (full random start).

The recipe (actor-driven discrete SAC):
  - 4x512 double-Q critic + categorical actor.
  - HER (relabel to achieved goals) — this is the curriculum; no spawn/reach curriculum.
  - Stable critic: avg target + Q-clip + hard target sync @1000 + alpha=0.01 (arXiv
    2209.10081). The critic only has to feed the actor a gradient, not be argmax-reachable.
  - Behaviour = SAMPLE the stochastic actor (agent.sample_actor_action). A sampled soft
    policy avoids the deterministic A<->B oscillation that argmax-over-critic (= DQN) falls
    into; the actor's entropy is the exploration — no epsilon-greedy.
  - max_steps=1000: long trajectories give HER rich relabel data and let the agent
    traverse to far random-start goals.
"""
import gymnasium as gym
import homebot  # noqa: F401  (registers HomeBot2D-Goal-V1)

from agent import SACAgent

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=1,
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,
)

agent = SACAgent(
    env=env,
    max_buffer_size=200000,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,    # Static 0.1 (Robert's best-success value; auto-tune never converged
                  # usefully here). alpha=0.01 was only ever needed to keep the critic's
                  # ARGMAX reachable for the OLD argmax-over-critic behaviour — obsolete now
                  # that behaviour SAMPLES the actor. At 0.01 the actor's entropy collapsed
                  # to ~0 by ep59 (run 377) -> deterministic actor -> the same wobble/
                  # oscillation SAC exists to avoid. With actor-driven behaviour the entropy
                  # is the FEATURE (exploration + anti-oscillation), so alpha goes back up.
    lr=3e-4,
    goal_noise_std=30.0,
    head_layers=4,
    head_hidden=512,
)

agent.train(episodes=2000, batch_size=64, warmup_steps=5000)
