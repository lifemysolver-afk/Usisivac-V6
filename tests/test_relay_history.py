import json
import os
import tempfile
from pathlib import Path
import pytest
import relay.triway_relay
from relay.triway_relay import get_history


def test_get_history_empty_and_nonexistent(tmp_path):
    non_existent = tmp_path / "non_existent.jsonl"
    relay.triway_relay.CHAT_LOG = non_existent
    assert get_history(limit=10) == []

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    relay.triway_relay.CHAT_LOG = empty_file
    assert get_history(limit=10) == []


def test_get_history_basic_and_participant_filter(tmp_path):
    log_file = tmp_path / "agent_conversation.jsonl"
    entries = []
    for i in range(100):
        sender = "Claude" if i % 2 == 0 else "Gemini"
        recipient = "Cline" if i % 3 == 0 else "User"
        entries.append({
            "from": sender,
            "to": recipient,
            "message": f"message_{i}",
            "index": i
        })

    with open(log_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    relay.triway_relay.CHAT_LOG = log_file

    # Test limit smaller than total entries
    recent_10 = get_history(limit=10)
    assert len(recent_10) == 10
    assert recent_10[-1]["index"] == 99
    assert recent_10[0]["index"] == 90

    # Test limit larger than total entries
    all_entries = get_history(limit=200)
    assert len(all_entries) == 100
    assert all_entries[0]["index"] == 0
    assert all_entries[-1]["index"] == 99

    # Test filtering by participant (case-insensitive)
    claude_msgs = get_history(limit=10, participant="claude")
    assert len(claude_msgs) == 10
    assert all(m["from"].lower() == "claude" or m["to"].lower() == "claude" for m in claude_msgs)
    assert claude_msgs[-1]["index"] == 98  # last even index <= 99
