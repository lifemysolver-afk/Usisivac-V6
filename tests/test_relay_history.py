import json
import pytest
from pathlib import Path
from relay import triway_relay

def test_get_history_empty_and_missing_file(tmp_path, monkeypatch):
    non_existent = tmp_path / "non_existent.jsonl"
    monkeypatch.setattr(triway_relay, "CHAT_LOG", non_existent)
    assert triway_relay.get_history() == []

    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()
    monkeypatch.setattr(triway_relay, "CHAT_LOG", empty_file)
    assert triway_relay.get_history() == []


def test_get_history_limit_and_order(tmp_path, monkeypatch):
    log_file = tmp_path / "agent_conversation.jsonl"
    entries = [
        {"from": "claude", "to": "gemini", "message": f"msg {i}"}
        for i in range(100)
    ]
    log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    history = triway_relay.get_history(limit=10)
    assert len(history) == 10
    assert history[0]["message"] == "msg 90"
    assert history[-1]["message"] == "msg 99"


def test_get_history_participant_filtering(tmp_path, monkeypatch):
    log_file = tmp_path / "agent_conversation.jsonl"
    entries = [
        {"from": "claude", "to": "gemini", "message": "msg 1"},
        {"from": "gemini", "to": "cline", "message": "msg 2"},
        {"from": "cline", "to": "claude", "message": "msg 3"},
        {"from": "gemini", "to": "claude", "message": "msg 4"},
    ]
    log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    cline_history = triway_relay.get_history(limit=10, participant="cline")
    assert len(cline_history) == 2
    assert [m["message"] for m in cline_history] == ["msg 2", "msg 3"]


def test_get_history_multibyte_utf8(tmp_path, monkeypatch):
    log_file = tmp_path / "agent_conversation.jsonl"
    entries = [
        {"from": "claude", "to": "gemini", "message": "Provera čćšđž 1"},
        {"from": "gemini", "to": "cline", "message": "Provera čćšđž 2"},
        {"from": "cline", "to": "claude", "message": "Provera čćšđž 3"},
    ]
    log_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    # Use a small chunk_size to force splitting across multi-byte UTF-8 boundaries
    history = triway_relay.get_history(limit=2, chunk_size=16)
    assert len(history) == 2
    assert history[0]["message"] == "Provera čćšđž 2"
    assert history[1]["message"] == "Provera čćšđž 3"
