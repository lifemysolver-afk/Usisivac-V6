import json
import pytest
from pathlib import Path
from relay import triway_relay

def test_get_history_empty_and_nonexistent(tmp_path, monkeypatch):
    non_existent = tmp_path / "does_not_exist.jsonl"
    monkeypatch.setattr(triway_relay, "CHAT_LOG", non_existent)
    assert triway_relay.get_history(limit=10) == []
    assert triway_relay.get_history(limit=0) == []
    assert triway_relay.get_history(limit=-5) == []

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", empty_file)
    assert triway_relay.get_history(limit=10) == []

def test_get_history_ordering_and_limit(tmp_path, monkeypatch):
    log_file = tmp_path / "chat.jsonl"
    messages = [
        {"timestamp": f"2026-01-01T00:00:{i:02d}", "from": "claude", "to": "gemini", "message": f"msg {i}"}
        for i in range(20)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    # Request limit=5, should return last 5 messages in chronological order
    history = triway_relay.get_history(limit=5)
    assert len(history) == 5
    assert [m["message"] for m in history] == [f"msg {i}" for i in range(15, 20)]

    # Request limit larger than total
    history_all = triway_relay.get_history(limit=50)
    assert len(history_all) == 20
    assert history_all == messages

def test_get_history_participant_filtering(tmp_path, monkeypatch):
    log_file = tmp_path / "chat.jsonl"
    messages = [
        {"from": "claude", "to": "gemini", "message": "msg 1"},
        {"from": "cline", "to": "gemini", "message": "msg 2"},
        {"from": "gemini", "to": "claude", "message": "msg 3"},
        {"from": "cline", "to": "claude", "message": "msg 4"},
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    gemini_history = triway_relay.get_history(limit=10, participant="gemini")
    assert len(gemini_history) == 3
    assert [m["message"] for m in gemini_history] == ["msg 1", "msg 2", "msg 3"]

    cline_history = triway_relay.get_history(limit=1, participant="cline")
    assert len(cline_history) == 1
    assert cline_history[0]["message"] == "msg 4"

def test_get_history_multibyte_and_chunk_boundary(tmp_path, monkeypatch):
    log_file = tmp_path / "chat.jsonl"
    # Create messages with unicode characters (emojis, Cyrillic) and long messages
    unicode_msg = "Pozdrav svima ⚡🚀 — testiranje karaktera čćžšđ"
    long_msg = "x" * 10000  # Longer than chunk size (8192 bytes)
    messages = [
        {"from": "claude", "to": "gemini", "message": f"{unicode_msg} {i}"}
        for i in range(50)
    ]
    messages.append({"from": "claude", "to": "cline", "message": long_msg})
    messages.append({"from": "cline", "to": "gemini", "message": "after long message"})

    with open(log_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    history = triway_relay.get_history(limit=3)
    assert len(history) == 3
    assert history[-1]["message"] == "after long message"
    assert history[-2]["message"] == long_msg
    assert history[-3]["message"] == f"{unicode_msg} 49"
