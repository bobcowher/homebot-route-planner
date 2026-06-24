"""The single tool the LLM drives: go_to(destination). Holds one live navigator
episode (model, env, obs, motion, pose) so pose persists across calls — exactly
the chained-eval setup, one leg per call."""
import gymnasium as gym
import torch

import homebot  # noqa: F401  (env registration)
from homebot.goals import GOAL_THRESHOLD
from evaluate import load_q_model, process_observation
from goal_geometry import distance, eval_step_budget
from motion import MotionState
from chained_eval import run_leg, REACH_OVERRIDE
from task_chain import leg_succeeded
from planner.world_model import WorldModel, DEST_TO_ENV


class NavigatorTool:
    def __init__(self, checkpoint="checkpoints/run325_q_model_best.pt",
                 readout="softmax_rel", temp=0.1, device=None,
                 render_mode="rgb_array", head_norm=False):
        # render_mode="human" opens a window and auto-shows every step (the env's
        # _get_obs draws to the window in human mode) -- used by the chat REPL so
        # you can watch the robot drive. Default "rgb_array" stays headless for
        # eval/smoke/tests.
        # head_norm=True is required for LayerNorm checkpoints (the macro-action runs);
        # macro_h is auto-detected from the checkpoint meta by load_q_model, and
        # run_leg decodes/executes the macro -- so pointing checkpoint at a macro model
        # (+head_norm) drives multi-step rollouts with no other change.
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(
            "HomeBot2D-V1", render_mode=render_mode, action_mode="discrete",
            obs_resolution=(96, 96), n_trash=2, max_steps=20000,
            map_name="default", random_start=True)
        self.base = self.env.unwrapped
        self.world = WorldModel(self.env)
        self.model = load_q_model(
            checkpoint, self.env.action_space.n, self.device,
            goal_layers=2, head_layers=4, use_motion=True, head_norm=head_norm)
        self.readout, self.temp = readout, temp
        self.obs = None
        self.ms = None

    def reset(self, seed=None) -> dict:
        raw, _ = self.env.reset(seed=seed)
        self.obs = process_observation(raw)
        self.ms = MotionState(self.env.action_space.n)
        return self.world.state()

    def state(self) -> dict:
        """Delegate to the world model so callers (e.g. eval_planner.score_task)
        can read state via nav.state() without reaching into nav.world."""
        return self.world.state()

    def go_to(self, destination: str) -> dict:
        try:
            gx, gy = self.world.resolve(destination)
        except (KeyError, ValueError) as e:
            return {"reached": False, "error": str(e), "state": self.world.state()}
        env_name = DEST_TO_ENV[destination]
        reach = REACH_OVERRIDE.get(env_name, GOAL_THRESHOLD)
        before = self.world.state()
        r = self.base._robot
        budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy))))
        # run_leg returns (reached, steps, obs, positions); the positions trace is
        # only for the spin metric, so discard it here.
        arrived, steps, self.obs, _ = run_leg(
            self.model, self.env, self.base, self.obs, (gx, gy),
            budget, self.device, self.readout, self.temp, self.ms, reach)
        after = self.world.state()
        # "reached" = the task actually completed (state delta), not just that the
        # robot got near the coordinate. "arrived" exposes the raw proximity too.
        reached = leg_succeeded(env_name, before, after, arrived)
        return {"reached": bool(reached), "arrived": bool(arrived),
                "steps": int(steps), "state": after}
