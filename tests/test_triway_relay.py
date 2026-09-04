import os
import json
import tempfile
import time
from pathlib import Path

from relay import triway_relay


def test_get_history_missing_file(monkeypatch, tmp_path):
    """Test get_history when the chat log file does not exist."""
    missing_file = tmp_path / "non_existent.jsonl"
    monkeypatch.setattr(triway_relay, "CHAT_LOG", missing_file)
    assert triway_relay.get_history(limit=10) == []


def test_get_history_empty_file(monkeypatch, tmp_path):
    """Test get_history when the chat log file is empty."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", empty_file)
    assert triway_relay.get_history(limit=10) == []


def test_get_history_limit_and_filtering(monkeypatch, tmp_path):
    """Test get_history limit and participant filtering with reverse chunk reading."""
    log_file = tmp_path / "agent_conversation.jsonl"

    entries = []
    for i in range(100):
        sender = "claude" if i % 2 == 0 else "gemini"
        receiver = "gemini" if i % 2 == 0 else "cline"
        entry = {
            "timestamp": f"2026-03-31T12:{i%60:02d}:00",
            "from": sender,
            "to": receiver,
            "message": f"Message {i} content",
            "type": "text",
            "protocol": "triway_relay_v1"
        }
        entries.append(entry)

    log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    # Test limit 10 without participant filter
    history = triway_relay.get_history(limit=10)
    assert len(history) == 10
    assert history == entries[-10:]

    # Test filtering by participant "claude"
    claude_entries = [e for e in entries if e["from"] == "claude" or e["to"] == "claude"]
    claude_history = triway_relay.get_history(limit=15, participant="claude")
    assert len(claude_history) == 15
    assert claude_history == claude_entries[-15:]


def test_get_history_utf8_and_corrupt_lines(monkeypatch, tmp_path):
    """Test get_history with unicode characters and invalid JSON lines."""
    log_file = tmp_path / "corrupt_utf8.jsonl"

    valid_entry = {
        "timestamp": "2026-03-31T12:00:00",
        "from": "claude",
        "to": "gemini",
        "message": "Šaljemo poruku sa ćiriličnim i latiničnim kôdnim znakovima ⚡!",
        "type": "text",
    }

    lines = [
        "corrupt json line {",
        json.dumps(valid_entry, ensure_ascii=False),
        "",
        "   ",
        json.dumps({"from": "gemini", "to": "claude", "message": "Second valid message"}),
    ]

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(triway_relay, "CHAT_LOG", log_file)

    history = triway_relay.get_history(limit=10)
    assert len(history) == 2
    assert history[0]["message"] == "Šaljemo poruku sa ćiriličnim i latiničnim kôdnim znakovima ⚡!"
    assert history[1]["message"] == "Second valid message"


def test_get_history_performance_speedup(tmp_path):
    """Benchmark performance improvement of reverse chunk reading vs full file load."""
    log_file = tmp_path / "large_agent_conversation.jsonl"

    # Create 5000 lines
    lines = []
    for i in range(5000):
        entry = {
            "timestamp": f"2026-03-31T12:{i%60:02d}:00",
            "from": "claude" if i % 2 == 0 else "gemini",
            "to": "gemini" if i % 2 == 0 else "cline",
            "message": f"Message {i} with extended context and payload details for Trinity Protocol.",
            "type": "text",
        }
        lines.append(json.dumps(entry))

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Time original full file read logic
    t0 = time.perf_counter()
    for _ in range(20):
        messages_old = []
        for line in log_file.read_text("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                messages_old.append(msg)
            except Exception:
                continue
        res_old = messages_old[-10:]
    t1 = time.perf_counter()
    dur_old = (t1 - t0) / 20

    # Time new reverse chunk reading logic via triway_relay.get_history
    import relay.triway_relay as tr
    tr.CHAT_LOG = log_file

    t0 = time.perf_counter()
    for _ in range(20):
        res_new = tr.get_history(limit=10)
    t2 = time.perf_counter()
    dur_new = (t2 - t0) / 20

    assert res_old == res_new
    assert dur_new < dur_old
    speedup = dur_old / max(dur_new, 1e-9)
    print(f"\n[BENCHMARK] Old: {dur_old*1000:.2f}ms, New: {dur_new*1000:.2f}ms, Speedup: {speedup:.1f}x")
