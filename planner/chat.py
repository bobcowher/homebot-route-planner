"""Repeatable chat session over the planner. One persistent PlannerAgent
(conversation) + one persistent NavigatorTool (live world episode) survive across
turns, so context carries: "go to the fridge" then "now bring it to me".

The session talks through an injected (read, speak) I/O pair -- the voice seam.
Defaults are terminal input()/print(); swap in ASR for read and TTS for speak to
get audio with no change to the loop or the planner.

    conda activate sac-homebot && python planner/chat.py
    # or, via conda run, stdin must be forwarded:
    conda run --no-capture-output -n sac-homebot python planner/chat.py
"""
import argparse
import os
import string
import sys
from datetime import datetime

# Allow direct-script invocation: put the repo root on the path so the absolute
# `planner.` package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _terminal_read(prompt="you> "):
    """Default reader: a line from stdin, or None on EOF (Ctrl-D)."""
    try:
        return input(prompt)
    except EOFError:
        return None


# Natural farewells end the session, not just "exit" -- this is how people (and,
# soon, ASR) actually close a conversation.
QUIT_WORDS = {"quit", "exit", "bye", "goodbye", "goodnight", "good night"}
RESET_WORDS = {"reset", "new scene", "start over"}


def _command(line: str) -> str:
    """Normalize a line for command matching: lowercase, strip surrounding
    punctuation/whitespace. So 'Goodbye.', 'bye!', 'GOOD NIGHT.' all match."""
    return line.lower().strip(string.punctuation + string.whitespace)


def _format_trace(name, arguments, result) -> str:
    """One readable record of a tool call and what the robot returned (the state),
    for the console printout and the chat log."""
    arg = arguments.get("destination", arguments) if isinstance(arguments, dict) else arguments
    head = f"[tool] {name}({arg!r})"
    if "error" in result and "reached" not in result:  # rejected before running
        return f"{head} -> error: {result['error']}"
    parts = [f"reached={result.get('reached')}"]
    if "arrived" in result and result["arrived"] != result.get("reached"):
        parts.append(f"arrived={result['arrived']}")  # got near but task not done
    if "steps" in result:
        parts.append(f"steps={result['steps']}")
    if "error" in result:
        parts.append(f"error={result['error']!r}")
    line = f"{head} -> " + " ".join(parts)
    state = result.get("state")
    if state:
        line += "\n        state: " + " ".join(f"{k}={v}" for k, v in state.items())
    return line


class Transcript:
    """Appends timestamped lines to a chat-log file (line-buffered so it can be
    tailed live). Captures the whole session -- utterances, replies, tool calls,
    state -- on disk so it can be reviewed after the fact."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", buffering=1)

    def write(self, line):
        stamp = datetime.now().strftime("%H:%M:%S")
        for sub in str(line).splitlines() or [""]:
            self._f.write(f"[{stamp}] {sub}\n")

    def close(self):
        self._f.close()


class ChatSession:
    """Drives a conversation: read an utterance, run it through the agent, speak
    the reply. `read()` returns the next utterance (or None to end the session);
    `speak(text)` emits a reply. Both are injectable for tests and for audio."""

    def __init__(self, agent, nav, read=_terminal_read, speak=print, seed=0,
                 log=None):
        self.agent = agent
        self.nav = nav
        self.read = read
        self.speak = speak
        self.seed = seed
        # log(line) records the transcript (utterances + replies + markers); the
        # agent's trace logs tool calls into the same sink for in-order history.
        self.log = log or (lambda line: None)
        self._scene = 0  # bumped each reset so every fresh scene is distinct

    def _say(self, text):
        """Speak a reply AND record it in the transcript."""
        self.speak(text)
        self.log(f"ROBOT: {text}")

    def start(self):
        self.nav.reset(seed=self.seed)
        self.log("=== session start ===")
        self._say("Ready. What would you like me to do? "
                  "(say 'reset' for a fresh scene, 'goodbye' to stop)")
        got_input = False
        while True:
            line = self.read()
            if line is None:           # end of stream (EOF / closed audio)
                if not got_input:
                    # input() EOF'd before a single line -- almost always a
                    # non-interactive stdin (e.g. plain `conda run` swallows it).
                    print("(no input received -- stdin isn't connected. If you "
                          "launched with `conda run`, add --no-capture-output, "
                          "or `conda activate` the env and run python directly.)",
                          file=sys.stderr)
                break
            got_input = True
            line = line.strip()
            if not line:
                continue
            cmd = _command(line)
            if cmd in QUIT_WORDS:
                break
            if cmd in RESET_WORDS:
                self._reset()
                continue
            self.log(f"YOU: {line}")
            self._say(self.agent.handle_utterance(line))
        self._say("Goodbye.")
        self.log("=== session end ===")

    def _reset(self):
        self._scene += 1
        seed = self.seed + self._scene
        self.nav.reset(seed=seed)   # new world scene
        self.agent.reset()          # clear the conversation
        self.log(f"=== reset: new scene (seed {seed}) ===")
        self._say("Fresh scene. What would you like me to do?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-mode", default="human", choices=["human", "rgb_array"],
                   help="human opens a window so you can watch the robot drive")
    p.add_argument("--log-dir", default="logs", help="where to write the chat log")
    p.add_argument("--checkpoint", default="checkpoints/run325_q_model_best.pt",
                   help="navigator checkpoint to drive (e.g. a macro-action model)")
    p.add_argument("--head-norm", action="store_true",
                   help="required for LayerNorm checkpoints (the macro-action runs)")
    p.add_argument("--readout", default="softmax_rel",
                   choices=["greedy", "softmax", "softmax_rel"],
                   help="action readout; macro models work at greedy (greedy==deploy)")
    p.add_argument("--temp", type=float, default=0.1, help="readout temperature")
    args = p.parse_args()

    from planner.navigator_tool import NavigatorTool
    from planner.llm_client import LLMClient
    from planner.agent_loop import PlannerAgent

    log_path = os.path.join(args.log_dir,
                            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    transcript = Transcript(log_path)
    print(f"(logging this session to {log_path})")

    def trace(name, arguments, result):
        line = _format_trace(name, arguments, result)
        print(line)             # console printout (so you see tool calls + state)
        transcript.write(line)  # and the chat log

    nav = NavigatorTool(checkpoint=args.checkpoint, readout=args.readout,
                        temp=args.temp, render_mode=args.render_mode,
                        head_norm=args.head_norm)
    agent = PlannerAgent(LLMClient(base_url=args.base_url, model=args.model), nav,
                         trace=trace)
    try:
        ChatSession(agent, nav, seed=args.seed, log=transcript.write).start()
    finally:
        transcript.close()


if __name__ == "__main__":
    main()
