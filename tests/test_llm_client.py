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


def test_normalize_recovers_tagged_tool_call_from_content():
    # Qwen via ollama sometimes emits the tool call as TEXT (its native
    # <tool_call> format) instead of the structured field -- it must still drive
    # the robot, not get printed to chat.
    msg = SimpleNamespace(
        content='<tool_call>\n{"name": "go_to", "arguments": {"destination": "fridge"}}\n</tool_call>',
        tool_calls=None)
    out = LLMClient._normalize(msg)
    assert out["tool_calls"] == [
        {"id": "text-0", "name": "go_to", "arguments": {"destination": "fridge"}}]


def test_normalize_recovers_untagged_tool_call_from_content():
    # The exact leak observed: opening tag mangled, bare JSON + stray closing tag.
    msg = SimpleNamespace(
        content=' modne\n{"name": "go_to", "arguments": {"destination": "fridge"}}\n</tool_call>',
        tool_calls=None)
    out = LLMClient._normalize(msg)
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == "go_to"
    assert out["tool_calls"][0]["arguments"] == {"destination": "fridge"}


def test_normalize_plain_prose_is_not_mistaken_for_a_tool_call():
    msg = SimpleNamespace(content="I've delivered your drink. Anything else?",
                          tool_calls=None)
    out = LLMClient._normalize(msg)
    assert out["tool_calls"] == []
    assert out["text"] == "I've delivered your drink. Anything else?"
