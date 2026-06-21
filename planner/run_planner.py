"""Run a single utterance end-to-end in sim against the real local LLM.

    python planner/run_planner.py --utterance "bring me a drink"
"""
import argparse
import os
import sys

# Allow direct-script invocation (python planner/run_planner.py): put the repo
# root on the path so the absolute `planner.` package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--utterance", required=True)
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from planner.navigator_tool import NavigatorTool
    from planner.llm_client import LLMClient
    from planner.agent_loop import PlannerAgent

    nav = NavigatorTool()
    nav.reset(seed=args.seed)
    agent = PlannerAgent(LLMClient(base_url=args.base_url, model=args.model), nav)
    response = agent.handle_utterance(args.utterance)
    print(f"\nROBOT: {response}")
    print(f"STATE: {nav.state()}")


if __name__ == "__main__":
    main()
