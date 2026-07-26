"""DeepSeek V4 tool-call history compatibility for recovery turns."""

from __future__ import annotations

from minisweagent.models.litellm_model import LitellmModel


class ReasoningReplayLitellmModel(LitellmModel):
    """Preserve reasoning history and normalize tool-call assistant content.

    LiteLLM already keeps ``reasoning_content`` when the response message is
    serialized. DeepSeek V4 additionally requires a non-null assistant content
    field on tool-call history. This adapter is instantiated only for recovery,
    after Gate G0 has failed.
    """

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        for message in prepared:
            if message.get("role") == "assistant" and message.get("tool_calls") and message.get("content") is None:
                message["content"] = ""
        return prepared
