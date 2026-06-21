"""Symbolic world model (perception stub) over the sim env.

The seam where real perception drops in later: today it reads the known map and
TaskManager state; tomorrow a VLM populates the same interface. The planner only
ever sees friendly destination names + task state, never pixels.
"""
from task_chain import resolve_goal

# Friendly destination name -> env goal-registry name.
DEST_TO_ENV = {
    "fridge": "go_to_fridge",
    "human": "go_to_human",
    "door": "go_to_door",
    "trash": "collect_trash",
}


class WorldModel:
    def __init__(self, env):
        self.env = env
        self.base = env.unwrapped

    def list_destinations(self) -> list[str]:
        """Currently reachable named targets. Trash only while some remains."""
        dests = ["fridge", "human", "door"]
        if self.base._task_manager.trash_positions:
            dests.append("trash")
        return dests

    def state(self) -> dict:
        base = self.base
        info = base._task_manager.get_info(base._robot)
        return {
            "carrying": base._robot.carrying,
            "trash_remaining": info["trash_remaining"],
            "drink_delivered": info["drink_delivered"],
            "package_delivered": info["package_delivered"],
            "robot_xy": (float(base._robot.x), float(base._robot.y)),
        }

    def resolve(self, name: str) -> tuple[float, float]:
        """Friendly name -> pixel (x, y). Raises KeyError on an unknown name;
        may raise ValueError if the target is currently unavailable (e.g. trash
        gone) — both are surfaced to the LLM by NavigatorTool as a tool error."""
        if name not in DEST_TO_ENV:
            raise KeyError(f"unknown destination {name!r}; valid: {sorted(DEST_TO_ENV)}")
        return resolve_goal(self.base, DEST_TO_ENV[name])
