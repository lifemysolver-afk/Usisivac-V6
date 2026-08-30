"""
Unit tests for get_history optimization in relay/triway_relay.py
"""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from relay.triway_relay import get_history, send

def test_get_history_empty_file():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)

    with patch("relay.triway_relay.CHAT_LOG", tmp_path):
        res = get_history(limit=10)
        assert res == []

    if tmp_path.exists():
        tmp_path.unlink()

def test_get_history_ordering_and_filtering():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)
        for i in range(100):
            entry = {
                "timestamp": f"2025-01-01T00:00:{i:02d}",
                "from": "claude" if i % 2 == 0 else "gemini",
                "to": "cline",
                "message": f"Message {i}",
                "type": "text"
            }
            tmp.write(json.dumps(entry) + "\n")

    with patch("relay.triway_relay.CHAT_LOG", tmp_path):
        # 1. Fetch top 10 overall
        h10 = get_history(limit=10)
        assert len(h10) == 10
        assert h10[-1]["message"] == "Message 99"
        assert h10[0]["message"] == "Message 90"

        # 2. Fetch top 5 for participant 'gemini' (odd indices: 99, 97, 95, 93, 91)
        h_gemini = get_history(limit=5, participant="gemini")
        assert len(h_gemini) == 5
        assert h_gemini[-1]["message"] == "Message 99"
        assert h_gemini[0]["message"] == "Message 91"

        # 3. Limit greater than total matching entries
        h_all = get_history(limit=500)
        assert len(h_all) == 100
        assert h_all[0]["message"] == "Message 0"
        assert h_all[-1]["message"] == "Message 99"

    if tmp_path.exists():
        tmp_path.unlink()
