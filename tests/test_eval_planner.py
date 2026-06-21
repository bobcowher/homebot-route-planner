from planner.eval_planner import score_task


class _Agent:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.said = []

    def handle_utterance(self, text):
        self.said.append(text)
        return self.reply


class _Nav:
    """Fakes a delivered-drink end state after reset."""
    def __init__(self):
        self._state = {"drink_delivered": False}

    def reset(self, seed=None):
        self._state = {"drink_delivered": True}  # pretend the agent succeeded
        return self._state

    def state(self):
        return self._state


def test_score_task_runs_utterance_and_checks_final_state():
    agent, nav = _Agent(), _Nav()
    ok = score_task(agent, nav, "bring me a drink",
                    check=lambda s: s["drink_delivered"], seed=0)
    assert ok is True
    assert agent.said == ["bring me a drink"]


def test_score_task_fails_when_check_false():
    agent, nav = _Agent(), _Nav()
    ok = score_task(agent, nav, "x", check=lambda s: s.get("nope", False), seed=0)
    assert ok is False
