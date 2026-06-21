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
from planner.world_model import WorldModel, DEST_TO_ENV


class NavigatorTool:
    def __init__(self, checkpoint="checkpoints/run314_q_model_best.pt",
                 readout="softmax_rel", temp=0.1, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(
            "HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
            obs_resolution=(96, 96), n_trash=2, max_steps=20000,
            map_name="default", random_start=True)
        self.base = self.env.unwrapped
        self.world = WorldModel(self.env)
        self.model = load_q_model(
            checkpoint, self.env.action_space.n, self.device,
            goal_layers=2, head_layers=4, use_motion=True)
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
        r = self.base._robot
        budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy))))
        reached, steps, self.obs = run_leg(
            self.model, self.env, self.base, self.obs, (gx, gy),
            budget, self.device, self.readout, self.temp, self.ms, reach)
        return {"reached": bool(reached), "steps": int(steps),
                "state": self.world.state()}
