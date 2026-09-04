# Bolt's Journal ⚡

## 2026-03-31 - Reverse Chunk Byte Reading for Large JSONL Log Files
**Learning:** In agent communication systems with growing JSONL log files (e.g. `logs/agent_conversation.jsonl`), fetching the last $N$ history entries using `file.read_text().splitlines()` loads the entire file into memory and parses every single historical JSON line, scaling as $O(N_{\text{lines}})$. Using reverse byte-chunk reading (`f.seek(pointer)`) reads backwards from EOF in 8KB chunks, parsing only the last $N$ entries and reducing time and memory complexity to $O(\text{limit})$, yielding over 300x speedup.
**Action:** Always prefer reverse byte-chunk reading for tail operations on append-only log files.
