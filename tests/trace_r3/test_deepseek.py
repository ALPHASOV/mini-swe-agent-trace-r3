from minisweagent.trace_r3.deepseek import ReasoningReplayLitellmModel


def test_recovery_model_preserves_reasoning_and_normalizes_tool_content():
    model = ReasoningReplayLitellmModel(
        model_name="deepseek/deepseek-v4-flash",
        cost_tracking="ignore_errors",
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "private reasoning replay token",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            "extra": {"cost": 0.1},
        }
    ]

    prepared = model._prepare_messages_for_api(messages)

    assert prepared[0]["content"] == ""
    assert prepared[0]["reasoning_content"] == "private reasoning replay token"
    assert "extra" not in prepared[0]
