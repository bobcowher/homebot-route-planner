"""Headless end-to-end smoke for the chat navigator (no LLM): build the real
NavigatorTool (loads the run314 champion), reset a scene, and drive go_to on each
destination. Confirms the model loads, run_leg's 4-tuple unpack works, and legs run
without crashing -- the plumbing chat.py drives, minus the qwen/ollama layer."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.navigator_tool import NavigatorTool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/run325_q_model_best.pt")
    ap.add_argument("--head-norm", action="store_true")
    ap.add_argument("--readout", default="softmax_rel")
    a = ap.parse_args()
    nav = NavigatorTool(checkpoint=a.checkpoint, readout=a.readout,
                        render_mode="rgb_array", head_norm=a.head_norm)  # headless
    nav.reset(seed=0)
    for dest in ["fridge", "human", "door"]:
        r = nav.go_to(dest)
        assert "reached" in r and "steps" in r, r
        print(f"go_to({dest!r}) -> reached={r['reached']} arrived={r.get('arrived')} "
              f"steps={r['steps']}")
    print("OK | navigator drove fridge/human/door without crashing")


if __name__ == "__main__":
    main()
