from planner.agent_loop import PlannerAgent


class MockLLM:
    """Returns scripted normalized responses, one per chat() call."""
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.scripted.pop(0)


class MockNav:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.visited = []

    def go_to(self, destination):
        self.visited.append(destination)
        return {"reached": self.outcomes.get(destination, True),
                "steps": 1, "state": {"carrying": None}}


def _tool(dest, cid="c1"):
    return {"tool_calls": [{"id": cid, "name": "go_to",
                            "arguments": {"destination": dest}}], "text": None}


def _say(text):
    return {"tool_calls": [], "text": text}


def _bad_tool(arguments, cid="c1"):
    """A go_to tool call with malformed arguments (what a real LLM sometimes emits)."""
    return {"tool_calls": [{"id": cid, "name": "go_to",
                            "arguments": arguments}], "text": None}


def test_executes_tool_calls_then_returns_spoken_response():
    llm = MockLLM([_tool("trash"), _tool("fridge"), _tool("human"), _say("Done.")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    out = agent.handle_utterance("tidy up and bring me a drink")
    assert nav.visited == ["trash", "fridge", "human"]
    assert out == "Done."


def test_tool_results_are_fed_back_into_conversation():
    llm = MockLLM([_tool("fridge"), _say("ok")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    agent.handle_utterance("get a drink")
    roles = [m["role"] for m in agent.conversation]
    assert "tool" in roles  # the go_to result was appended for the LLM to see


def test_conversation_persists_across_utterances():
    llm = MockLLM([_say("hi"), _say("bye")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    agent.handle_utterance("hello")
    agent.handle_utterance("later")
    user_turns = [m for m in agent.conversation if m["role"] == "user"]
    assert len(user_turns) == 2


def test_missing_destination_does_not_crash_loop():
    # A real LLM sometimes emits go_to with no destination. The loop must feed an
    # error result back (so the model can recover) instead of crashing the run.
    llm = MockLLM([_bad_tool({}), _say("Sorry, I got confused.")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    out = agent.handle_utterance("do the thing")
    assert nav.visited == []  # malformed call never reaches the navigator
    tool_msgs = [m for m in agent.conversation if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and "error" in tool_msgs[0]["content"]
    assert out == "Sorry, I got confused."


def test_tool_call_budget_stops_infinite_loops():
    llm = MockLLM([_tool("fridge")] * 50)  # never says a final message
    nav = MockNav()
    agent = PlannerAgent(llm, nav, max_tool_calls=12)
    out = agent.handle_utterance("loop forever")
    assert len(nav.visited) == 12
    assert "couldn't" in out.lower() or "could not" in out.lower()
