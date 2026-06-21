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


class ChatSession:
    """Drives a conversation: read an utterance, run it through the agent, speak
    the reply. `read()` returns the next utterance (or None to end the session);
    `speak(text)` emits a reply. Both are injectable for tests and for audio."""

    def __init__(self, agent, nav, read=_terminal_read, speak=print, seed=0):
        self.agent = agent
        self.nav = nav
        self.read = read
        self.speak = speak
        self.seed = seed
        self._scene = 0  # bumped each reset so every fresh scene is distinct

    def start(self):
        self.nav.reset(seed=self.seed)
        self.speak("Ready. What would you like me to do? "
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
            self.speak(self.agent.handle_utterance(line))
        self.speak("Goodbye.")

    def _reset(self):
        self._scene += 1
        self.nav.reset(seed=self.seed + self._scene)  # new world scene
        self.agent.reset()                            # clear the conversation
        self.speak("Fresh scene. What would you like me to do?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from planner.navigator_tool import NavigatorTool
    from planner.llm_client import LLMClient
    from planner.agent_loop import PlannerAgent

    nav = NavigatorTool()
    agent = PlannerAgent(LLMClient(base_url=args.base_url, model=args.model), nav)
    ChatSession(agent, nav, seed=args.seed).start()


if __name__ == "__main__":
    main()
