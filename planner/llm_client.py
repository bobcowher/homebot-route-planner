"""Thin wrapper over a local OpenAI-compatible endpoint (ollama/vLLM on the 5090).
Endpoint-agnostic: base_url + model are configurable. Normalizes responses into a
small shape the agent loop consumes, so the loop can be tested with a mock."""
import json

SYSTEM_PROMPT = """You are the planner for a home robot. You accomplish tasks by \
moving the robot between named destinations with the go_to tool. You cannot do \
anything except call go_to and talk to the user.

Destinations: fridge, human, door, trash.

How the world works:
- go_to("trash") collects the trash there (use it to tidy up).
- go_to("fridge") picks up a drink.
- go_to("human") while carrying a drink delivers it to the person.
- go_to("door") picks up a package waiting at the door; go_to("human") while \
carrying it delivers it.

After each go_to you get back whether the robot reached the destination and the \
current state (what it is carrying, trash remaining, deliveries done). If a \
go_to times out (reached=false), decide whether to retry, try a different route, \
or tell the user you could not complete the task. When the task is finished, or \
cannot be done, reply with a short natural-language message to the user instead \
of calling a tool."""

GO_TO_TOOL = {
    "type": "function",
    "function": {
        "name": "go_to",
        "description": "Drive the robot to a named destination. Returns whether "
                       "it reached the destination and the updated world state.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": ["fridge", "human", "door", "trash"],
                    "description": "Where to send the robot.",
                },
            },
            "required": ["destination"],
        },
    },
}


class LLMClient:
    def __init__(self, base_url="http://localhost:11434/v1",
                 model="qwen2.5:14b-instruct", api_key="local"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @staticmethod
    def _normalize(message) -> dict:
        """OpenAI chat message -> {"tool_calls": [...], "text": str|None}."""
        tool_calls = []
        for tc in (message.tool_calls or []):
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments or "{}"),
            })
        return {"tool_calls": tool_calls,
                "text": None if tool_calls else message.content}

    def chat(self, messages: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=[GO_TO_TOOL])
        return self._normalize(resp.choices[0].message)
