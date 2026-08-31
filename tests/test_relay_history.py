import os
import json
import tempfile
import pytest
from unittest.mock import patch
from relay.triway_relay import get_history


def test_get_history_nonexistent_file():
    """Returns empty list if log file does not exist."""
    with patch("relay.triway_relay.CHAT_LOG") as mock_path:
        mock_path.exists.return_value = False
        assert get_history(limit=10) == []


def test_get_history_empty_file():
    """Handles empty log file correctly."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
        tf_path = tf.name

    try:
        from pathlib import Path
        with patch("relay.triway_relay.CHAT_LOG", Path(tf_path)):
            assert get_history(limit=10) == []
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)


def test_get_history_small_log():
    """Retrieves all items when total count is less than limit."""
    entries = [
        {"timestamp": "2026-05-01T10:00:00", "from": "claude", "to": "cline", "message": "msg 1"},
        {"timestamp": "2026-05-01T10:01:00", "from": "gemini", "to": "claude", "message": "msg 2"},
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tf:
        for entry in entries:
            tf.write(json.dumps(entry) + "\n")
        tf_path = tf.name

    try:
        from pathlib import Path
        with patch("relay.triway_relay.CHAT_LOG", Path(tf_path)):
            history = get_history(limit=10)
            assert history == entries
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)


def test_get_history_limit_and_chunk_boundary():
    """Correctly parses and limits logs across chunk boundaries."""
    entries = [
        {"timestamp": f"2026-05-01T10:00:{i:02d}", "from": "claude" if i % 2 == 0 else "gemini", "to": "cline", "message": f"Message contents {i} " + "x" * 100}
        for i in range(100)
    ]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tf:
        for entry in entries:
            tf.write(json.dumps(entry) + "\n")
        tf_path = tf.name

    try:
        from pathlib import Path
        with patch("relay.triway_relay.CHAT_LOG", Path(tf_path)):
            # Test with small chunk size to trigger multi-chunk reading
            history = get_history(limit=15, chunk_size=256)
            assert len(history) == 15
            assert history == entries[-15:]

            # Test with participant filtering
            history_claude = get_history(limit=10, participant="claude", chunk_size=256)
            expected_claude = [e for e in entries if e["from"] == "claude" or e["to"] == "claude"][-10:]
            assert len(history_claude) == 10
            assert history_claude == expected_claude
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)
