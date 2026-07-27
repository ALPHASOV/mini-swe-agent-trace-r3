"""DeepSeek V4 reasoning and tool-call history compatibility."""

from __future__ import annotations

from minisweagent.models.litellm_model import LitellmModel


class ReasoningReplayLitellmModel(LitellmModel):
    """Preserve reasoning history and normalize tool-call assistant content.

    LiteLLM already keeps ``reasoning_content`` when the response message is
    serialized. DeepSeek V4 additionally requires a non-null assistant content
    field on tool-call history. TRACE-R³ uses this adapter for the independent
    read-only validation-planning conversation and for recovery after Gate G0.
    """

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        for message in prepared:
            if message.get("role") == "assistant" and message.get("tool_calls") and message.get("content") is None:
                message["content"] = ""
        return prepared
