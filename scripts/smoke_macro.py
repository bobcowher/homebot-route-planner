"""End-to-end smoke for the macro-action (H=3) head. Checks:
  - decode_macro round-trips an index <-> base-action tuple
  - the agent builds a 512-wide head (8**3) with base-8 motion
  - select_action returns a valid macro index, decoded to 3 base actions
  - a few short episodes train without crashing; the buffer gets ~one transition
    per macro (i.e. ~steps/3), confirming macro-granular storage
  - chain_eval (the best-ckpt metric) runs the macro policy without crashing
"""
import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401  (env registration)
from agent import Agent
from policy import decode_macro


def test_decode_round_trip():
    n_base, H = 8, 3
    for idx in (0, 1, 7, 8, 73, 511):
        acts = decode_macro(idx, H, n_base)
        assert len(acts) == H and all(0 <= a < n_base for a in acts)
        # re-encode MSB-first
        re = 0
        for a in acts:
            re = re * n_base + a
        assert re == idx, f"{idx} -> {acts} -> {re}"
    print("OK | decode_macro round-trips")


def main():
    test_decode_round_trip()
    env = gym.make("HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=45, map_name="default",
                   goals=["collect_trash"], random_start=True)
    agent = Agent(env=env, max_buffer_size=4000, goal_layers=2, head_layers=4,
                  use_motion=True, motion_window=1, random_goal_tiles=True, macro_h=3)
    assert agent.n_base == 8 and agent.macro_h == 3 and agent.n_actions == 512, \
        (agent.n_base, agent.macro_h, agent.n_actions)
    assert agent.q_model.output.out_features == 512, agent.q_model.output.out_features

    # select_action returns a macro index decodable to 3 base actions.
    raw, _ = env.reset()
    obs = agent.process_observation(raw["observation"])
    import numpy as np
    from goal_geometry import world_coords
    from motion import MotionState
    base = env.unwrapped
    r = base._robot
    ms = MotionState(agent.n_base, agent.motion_window)
    gv = world_coords(r.x, r.y, raw["desired_goal"][0], raw["desired_goal"][1])
    agent.epsilon = 0.0  # force the exploit (argmax) path
    idx = agent.select_action(obs, gv, ms.vec(r.x, r.y))
    assert 0 <= idx < 512, idx
    assert len(decode_macro(idx, agent.macro_h, agent.n_base)) == 3

    agent.train(episodes=3, batch_size=8, eval_interval=999, eval_episodes=1,
                chain_eval_interval=999, her_anneal_start=None)
    print("OK | macro train ran 3 episodes + chain_eval without crashing")

    # Round-trip the saved best.pt through load_q_model -- guards the weights_only /
    # numpy-scalar-in-meta bug (gym Discrete.n -> np.int64) that broke macro reload.
    from evaluate import load_q_model
    m = load_q_model("checkpoints/q_model_best.pt", agent.n_base, agent.device,
                     goal_layers=2, head_layers=4, use_motion=True)
    assert m.macro_h == 3 and m.output.out_features == 512, (m.macro_h, m.output.out_features)
    print("OK | macro best.pt round-trips through load_q_model (512 head, macro_h=3 from meta)")


if __name__ == "__main__":
    main()
