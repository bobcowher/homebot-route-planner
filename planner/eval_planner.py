"""NL-task suite for the planner: each task is (utterance, check(state)->bool).
Run the real LLM end-to-end in sim and score on final world state — the
chained_eval philosophy with the LLM generating the chain. score_task is
LLM/Nav-agnostic (duck-typed) so it is unit-testable with fakes.

    python planner/eval_planner.py --episodes 10
"""
import argparse
import os
import sys

# Allow direct-script invocation: put the repo root on the path so the absolute
# `planner.` package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS = [
    ("Please tidy up — clear the trash.",
     lambda s: s["trash_remaining"] == 0),
    ("Bring me a drink from the fridge.",
     lambda s: s["drink_delivered"]),
    ("Bring me the package from the door.",
     lambda s: s["package_delivered"]),
    ("Clear the trash and then bring me a drink.",
     lambda s: s["trash_remaining"] == 0 and s["drink_delivered"]),
]


def score_task(agent, nav, utterance, check, seed=0) -> bool:
    nav.reset(seed=seed)
    agent.handle_utterance(utterance)
    return bool(check(nav.state()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    from planner.navigator_tool import NavigatorTool
    from planner.llm_client import LLMClient
    from planner.agent_loop import PlannerAgent

    nav = NavigatorTool()
    llm = LLMClient(base_url=args.base_url, model=args.model)
    passed = 0
    total = 0
    for utterance, check in TASKS:
        task_passed = 0
        for i in range(args.episodes):
            agent = PlannerAgent(llm, nav)  # fresh conversation per task run
            ok = score_task(agent, nav, utterance, check, seed=i)
            task_passed += int(ok)
        passed += task_passed
        total += args.episodes
        print(f"  {utterance!r}: {task_passed}/{args.episodes} "
              f"= {100.0 * task_passed / args.episodes:.0f}%")
    print(f"\nPlanner task completion: {passed}/{total} = {100.0 * passed / total:.0f}%")


if __name__ == "__main__":
    main()
