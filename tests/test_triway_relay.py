import os
import json
import pytest
from pathlib import Path
from relay import triway_relay

def test_get_history_nonexistent_file(tmp_path, monkeypatch):
    test_log = tmp_path / "nonexistent.jsonl"
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)
    assert triway_relay.get_history(limit=10) == []
    assert triway_relay.get_history(limit=0) == []
    assert triway_relay.get_history(limit=-5) == []


def test_get_history_empty_file(tmp_path, monkeypatch):
    test_log = tmp_path / "empty.jsonl"
    test_log.write_text("", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)
    assert triway_relay.get_history(limit=10) == []


def test_get_history_single_message(tmp_path, monkeypatch):
    test_log = tmp_path / "single.jsonl"
    msg = {"timestamp": "2026-04-30T12:00:00", "from": "claude", "to": "gemini", "message": "hello", "type": "text"}
    test_log.write_text(json.dumps(msg) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)

    history = triway_relay.get_history(limit=5)
    assert len(history) == 1
    assert history[0]["message"] == "hello"

    # Filter participant
    assert len(triway_relay.get_history(limit=5, participant="claude")) == 1
    assert len(triway_relay.get_history(limit=5, participant="cline")) == 0


def test_get_history_fewer_than_limit(tmp_path, monkeypatch):
    test_log = tmp_path / "few.jsonl"
    messages = [
        {"from": "gemini", "to": "claude", "message": f"msg {i}"}
        for i in range(5)
    ]
    test_log.write_text("\n".join(json.dumps(m) for m in messages) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)

    res = triway_relay.get_history(limit=10)
    assert len(res) == 5
    assert [m["message"] for m in res] == [f"msg {i}" for i in range(5)]


def test_get_history_more_than_limit(tmp_path, monkeypatch):
    test_log = tmp_path / "many.jsonl"
    messages = [
        {"from": "gemini", "to": "claude", "message": f"msg {i}"}
        for i in range(50)
    ]
    test_log.write_text("\n".join(json.dumps(m) for m in messages) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)

    res = triway_relay.get_history(limit=10)
    assert len(res) == 10
    assert [m["message"] for m in res] == [f"msg {i}" for i in range(40, 50)]


def test_get_history_across_multiple_chunks(tmp_path, monkeypatch):
    # Write enough lines to exceed 8192 bytes chunk size
    test_log = tmp_path / "multi_chunk.jsonl"
    messages = [
        {"from": "claude" if i % 2 == 0 else "gemini",
         "to": "gemini" if i % 2 == 0 else "claude",
         "message": f"message with padding: {'x' * 200}_{i}"}
        for i in range(200)
    ]
    test_log.write_text("\n".join(json.dumps(m) for m in messages) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)

    # Fetch last 30 messages
    res = triway_relay.get_history(limit=30)
    assert len(res) == 30
    assert res[-1]["message"] == f"message with padding: {'x' * 200}_199"
    assert res[0]["message"] == f"message with padding: {'x' * 200}_170"

    # Filter with participant across chunks
    res_claude = triway_relay.get_history(limit=20, participant="claude")
    assert len(res_claude) == 20
    assert all(m["from"] == "claude" or m["to"] == "claude" for m in res_claude)


def test_get_history_with_corrupt_lines_and_blank_lines(tmp_path, monkeypatch):
    test_log = tmp_path / "corrupt.jsonl"
    content = (
        "corrupted non-json line 1\n\n"
        + json.dumps({"from": "claude", "to": "gemini", "message": "valid 1"}) + "\n\n"
        + "corrupted non-json line 2\n"
        + json.dumps({"from": "gemini", "to": "claude", "message": "valid 2"}) + "\n\n\n"
    )
    test_log.write_text(content, encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", test_log)

    res = triway_relay.get_history(limit=10)
    assert len(res) == 2
    assert res[0]["message"] == "valid 1"
    assert res[1]["message"] == "valid 2"
