"""Closed-loop ReAct planner. handle_utterance(text) -> spoken_response over a
persistent message history. Voice is adapters outside this class: ASR feeds text
in, TTS speaks the returned response."""
import json

from planner.llm_client import SYSTEM_PROMPT

MAX_TOOL_CALLS = 12


class PlannerAgent:
    def __init__(self, client, navigator, system_prompt=SYSTEM_PROMPT,
                 max_tool_calls=MAX_TOOL_CALLS):
        self.client = client
        self.nav = navigator
        self.max_tool_calls = max_tool_calls
        self.conversation = [{"role": "system", "content": system_prompt}]

    def handle_utterance(self, text: str) -> str:
        self.conversation.append({"role": "user", "content": text})
        for _ in range(self.max_tool_calls):
            resp = self.client.chat(self.conversation)
            if not resp["tool_calls"]:
                self.conversation.append(
                    {"role": "assistant", "content": resp["text"]})
                return resp["text"]
            # Record the assistant's tool-call turn, then each tool result.
            self.conversation.append({
                "role": "assistant",
                "content": resp.get("text"),
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for tc in resp["tool_calls"]],
            })
            for tc in resp["tool_calls"]:
                result = self.nav.go_to(tc["arguments"]["destination"])
                self.conversation.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(result)})
        msg = "I couldn't complete that within the step budget."
        self.conversation.append({"role": "assistant", "content": msg})
        return msg
