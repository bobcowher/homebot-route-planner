import json
from types import SimpleNamespace

from planner.llm_client import LLMClient, SYSTEM_PROMPT, GO_TO_TOOL


def test_go_to_tool_schema_shape():
    assert GO_TO_TOOL["type"] == "function"
    fn = GO_TO_TOOL["function"]
    assert fn["name"] == "go_to"
    assert "destination" in fn["parameters"]["properties"]


def test_system_prompt_teaches_mechanics_and_destinations():
    p = SYSTEM_PROMPT.lower()
    assert "go_to" in p
    for token in ("fridge", "human", "door", "trash"):
        assert token in p
    assert "drink" in p and "deliver" in p  # env mechanics are taught


def test_normalize_tool_call_message():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="go_to",
                                     arguments=json.dumps({"destination": "fridge"})))],
    )
    out = LLMClient._normalize(msg)
    assert out["text"] is None
    assert out["tool_calls"] == [
        {"id": "call_1", "name": "go_to", "arguments": {"destination": "fridge"}}]


def test_normalize_final_text_message():
    msg = SimpleNamespace(content="All done.", tool_calls=None)
    out = LLMClient._normalize(msg)
    assert out["tool_calls"] == [] and out["text"] == "All done."
