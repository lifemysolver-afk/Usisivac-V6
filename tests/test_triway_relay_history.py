import os
import json
import tempfile
import time
from unittest.mock import patch
import pytest

from relay import triway_relay


def test_get_history_non_existent_file(tmp_path):
    non_existent = tmp_path / "non_existent.jsonl"
    with patch("relay.triway_relay.CHAT_LOG", non_existent):
        assert triway_relay.get_history(limit=10) == []


def test_get_history_empty_file(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with patch("relay.triway_relay.CHAT_LOG", empty_file):
        assert triway_relay.get_history(limit=10) == []


def test_get_history_limit_zero_or_negative(tmp_path):
    log_file = tmp_path / "test.jsonl"
    log_file.write_text('{"from": "a", "to": "b", "message": "hello"}\n', encoding="utf-8")
    with patch("relay.triway_relay.CHAT_LOG", log_file):
        assert triway_relay.get_history(limit=0) == []
        assert triway_relay.get_history(limit=-5) == []


def test_get_history_small_file_and_participant_filter(tmp_path):
    log_file = tmp_path / "chat.jsonl"
    entries = [
        {"timestamp": "2025-01-01T00:00:00", "from": "claude", "to": "gemini", "message": "msg 1"},
        {"timestamp": "2025-01-01T00:00:01", "from": "gemini", "to": "cline", "message": "msg 2"},
        {"timestamp": "2025-01-01T00:00:02", "from": "cline", "to": "claude", "message": "msg 3"},
        {"timestamp": "2025-01-01T00:00:03", "from": "claude", "to": "cline", "message": "msg 4"},
    ]
    log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    with patch("relay.triway_relay.CHAT_LOG", log_file):
        # Fetch last 2
        all_last2 = triway_relay.get_history(limit=2)
        assert len(all_last2) == 2
        assert all_last2[0]["message"] == "msg 3"
        assert all_last2[1]["message"] == "msg 4"

        # Participant filter: 'claude'
        claude_msgs = triway_relay.get_history(limit=10, participant="claude")
        assert len(claude_msgs) == 3
        assert [m["message"] for m in claude_msgs] == ["msg 1", "msg 3", "msg 4"]

        # Case insensitive participant filter
        gemini_msgs = triway_relay.get_history(limit=10, participant="GEMINI")
        assert len(gemini_msgs) == 2
        assert [m["message"] for m in gemini_msgs] == ["msg 1", "msg 2"]


def test_get_history_large_file_and_invalid_lines(tmp_path):
    log_file = tmp_path / "large_chat.jsonl"
    lines = []
    # Create 5000 valid messages interspersed with corrupted lines
    for i in range(5000):
        if i % 100 == 0:
            lines.append("CORRUPTED_NON_JSON_LINE_{i}")
        p_from = "claude" if i % 2 == 0 else "gemini"
        p_to = "gemini" if i % 2 == 0 else "cline"
        msg = {
            "timestamp": "2025-01-01T00:00:00",
            "from": p_from,
            "to": p_to,
            "message": f"Message {i} with special UTF-8 chars 🚀 и ћирилица",
        }
        lines.append(json.dumps(msg, ensure_ascii=False))

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with patch("relay.triway_relay.CHAT_LOG", log_file):
        t0 = time.perf_counter()
        history = triway_relay.get_history(limit=50, participant="claude")
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        assert len(history) == 50
        assert elapsed_ms < 50.0  # Should be under 50ms (typically ~1ms)
        assert history[-1]["message"].startswith("Message 4998")

        context = triway_relay.get_context_for_agent("claude", max_messages=5)
        assert "Message 4998" in context
