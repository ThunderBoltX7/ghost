from unittest.mock import MagicMock, patch
from ghost_cli.context import (
    repair_messages,
    _compact_messages,
    _microcompact_messages,
    _fallback_trim,
)
from ghost_cli.config import HOLDER


def test_repair_messages_orphan_tool_dropped():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "tool", "tool_call_id": "orphan_1", "content": "dangling result"},
        {"role": "user", "content": "hello"},
    ]
    repaired = repair_messages(messages)
    assert len(repaired) == 2
    assert repaired[0]["role"] == "system"
    assert repaired[1]["role"] == "user"


def test_repair_messages_missing_tool_gets_synthetic():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "list_dir", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    ]
    repaired = repair_messages(messages)
    assert len(repaired) == 5
    assert repaired[3]["tool_call_id"] == "call_1"
    assert repaired[3]["content"] == "file contents"
    # call_2 was missing a tool result, should have synthetic entry inserted
    assert repaired[4]["role"] == "tool"
    assert repaired[4]["tool_call_id"] == "call_2"
    assert "interrupted" in repaired[4]["content"]


def test_repair_messages_valid_sequence_unchanged():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    repaired = repair_messages(messages)
    assert len(repaired) == 5
    assert repaired == messages


def test_compact_messages_structure():
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "step 1"},
        {"role": "assistant", "content": "ack 1"},
        {"role": "user", "content": "step 2"},
        {"role": "assistant", "content": "ack 2"},
        {"role": "user", "content": "step 3"},
        {"role": "assistant", "content": "ack 3"},
    ]

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Summary of steps 1 and 2."

    with patch.object(HOLDER, "client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_resp
        compacted = _compact_messages(messages)

    # System message always first
    assert compacted[0]["role"] == "system"
    # Second message is the summary
    assert compacted[1]["role"] == "user"
    assert "[Working state summary" in compacted[1]["content"]
    # Tail starts on a user message or standalone assistant message
    assert compacted[2]["role"] in ("user", "assistant")
    # No orphan tool messages
    for i, m in enumerate(compacted):
        if m.get("role") == "tool":
            assert i > 0 and compacted[i - 1].get("role") == "assistant"


def test_microcompact_messages():
    # Construct 12 tool messages
    messages = [{"role": "system", "content": "sys"}]
    for i in range(12):
        messages.append({"role": "user", "content": f"query {i}"})
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"tc_{i}", "function": {"name": "test_fn", "arguments": "{}"}}],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"tc_{i}",
            "content": f"Long tool result content for message number {i} " * 5,
        })

    micro = _microcompact_messages(messages)

    tool_msgs = [m for m in micro if m.get("role") == "tool"]
    assert len(tool_msgs) == 12

    # The first 4 tool results (12 - 8 = 4) should be stubbed with [cleared: ...]
    for m in tool_msgs[:4]:
        assert m["content"].startswith("[cleared:")
        assert "tc_" in m["tool_call_id"]

    # The last 8 tool results should remain untouched
    for m in tool_msgs[4:]:
        assert not m["content"].startswith("[cleared:")
        assert "Long tool result content" in m["content"]

    # Pairings intact (each tool message still has tool_call_id)
    assert len(micro) == len(messages)
