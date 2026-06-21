from planner.chat import ChatSession


class MockAgent:
    """Records utterances, returns a canned reply, supports reset()."""
    def __init__(self, reply="ok"):
        self.reply = reply
        self.utterances = []
        self.resets = 0

    def handle_utterance(self, text):
        self.utterances.append(text)
        return self.reply

    def reset(self):
        self.resets += 1


class MockNav:
    def __init__(self):
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return {}


def _scripted_reader(lines):
    """Yields each line in turn, then None (stream end) forever after."""
    it = iter(lines)

    def read(_prompt=""):
        return next(it, None)
    return read


def _session(lines, **kw):
    agent, nav = MockAgent(), MockNav()
    spoken = []
    s = ChatSession(agent, nav, read=_scripted_reader(lines),
                    speak=spoken.append, **kw)
    return s, agent, nav, spoken


def test_utterances_routed_to_agent_and_replies_spoken():
    s, agent, nav, spoken = _session(["clear the trash", "bring me a drink"])
    s.start()
    assert agent.utterances == ["clear the trash", "bring me a drink"]
    assert "ok" in spoken  # the agent reply was spoken


def test_start_resets_nav_to_a_fresh_scene():
    s, agent, nav, spoken = _session([], seed=7)
    s.start()
    assert nav.reset_seeds == [7]  # one fresh scene at session start


def test_quit_ends_the_loop():
    s, agent, nav, spoken = _session(["hello", "quit", "should not run"])
    s.start()
    assert agent.utterances == ["hello"]  # nothing after quit


def test_natural_farewells_end_the_loop_despite_case_and_punctuation():
    # People (and ASR) end conversations with "Goodbye.", "bye!", "Goodnight" --
    # not "exit". These must end the session, not get sent to the LLM.
    for farewell in ["Goodbye.", "bye!", "Goodnight", "GOOD NIGHT.", "  bye  "]:
        s, agent, nav, spoken = _session([farewell, "should not run"])
        s.start()
        assert agent.utterances == [], f"{farewell!r} leaked to the agent"


def test_end_of_stream_ends_the_loop():
    # read() returning None (EOF / closed audio stream) ends the session cleanly.
    s, agent, nav, spoken = _session(["hello"])
    s.start()
    assert agent.utterances == ["hello"]


def test_hint_when_stdin_is_dead_before_any_input(capsys):
    # `conda run` (no --no-capture-output) gives input() an instant EOF, so the
    # REPL exits before reading anything -- looks like it "died". Emit a hint.
    s, agent, nav, spoken = _session([])  # read() returns None immediately
    s.start()
    err = capsys.readouterr().err
    assert "--no-capture-output" in err


def test_no_dead_stdin_hint_after_normal_use(capsys):
    s, agent, nav, spoken = _session(["do a thing", "quit"])
    s.start()
    assert "--no-capture-output" not in capsys.readouterr().err


def test_blank_lines_are_skipped():
    s, agent, nav, spoken = _session(["", "   ", "real"])
    s.start()
    assert agent.utterances == ["real"]


def test_reset_clears_conversation_and_makes_new_scene():
    s, agent, nav, spoken = _session(["do a thing", "reset", "do another"], seed=0)
    s.start()
    assert agent.resets == 1                  # conversation cleared
    assert nav.reset_seeds == [0, 1]          # start scene + a new scene on reset
    assert agent.utterances == ["do a thing", "do another"]  # 'reset' not an utterance
