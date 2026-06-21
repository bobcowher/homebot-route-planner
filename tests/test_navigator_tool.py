from planner.navigator_tool import NavigatorTool


def test_reset_returns_state_and_go_to_returns_outcome_shape():
    nav = NavigatorTool()  # loads the 314 champion
    s = nav.reset(seed=0)
    assert "robot_xy" in s
    out = nav.go_to("human")
    assert {"reached", "steps", "state"} <= set(out)
    assert isinstance(out["reached"], bool) and isinstance(out["steps"], int)


def test_unknown_destination_is_error_not_crash():
    nav = NavigatorTool()
    nav.reset(seed=0)
    out = nav.go_to("bogus")
    assert out["reached"] is False and "error" in out


def test_state_delegates_to_world():
    nav = NavigatorTool()
    nav.reset(seed=0)
    assert nav.state() == nav.world.state()
    assert "trash_remaining" in nav.state()
