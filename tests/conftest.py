import sys
from unittest.mock import Mock
import numpy as np
import pytest

# ─── Global Mocks for CI ─────────────────────────────────────────────────────

# Mock SentenceTransformer to avoid downloading weights
class MockEmbedder:
    def encode(self, texts, **kwargs):
        # Return normalized random vectors
        if isinstance(texts, str):
            v = np.random.randn(384).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-9)
        v = np.random.randn(len(texts), 384).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

@pytest.fixture(autouse=True, scope="session")
def mock_session_deps():
    # Patch sentence_transformers
    mock_st = Mock()
    mock_st.SentenceTransformer.return_value = MockEmbedder()
    sys.modules["sentence_transformers"] = mock_st

    # Patch chromadb
    mock_chroma = Mock()
    sys.modules["chromadb"] = mock_chroma
    sys.modules["chromadb.utils"] = Mock()
    sys.modules["chromadb.utils.embedding_functions"] = Mock()

    # Patch other potential network/heavy dependencies
    sys.modules["google.genai"] = Mock()

    yield

@pytest.fixture(autouse=True)
def mock_filesystem_deps(tmp_path, monkeypatch):
    # Ensure ChromaDB and other files go to a temp directory
    from pathlib import Path

    # Mock paths in core.rag_engine
    # We need to import it here after mocks are set
    try:
        import core.rag_engine as rag
        monkeypatch.setattr(rag, "CHROMA_PATH", tmp_path / "chroma")
        monkeypatch.setattr(rag, "FALLBACK_DIR", tmp_path / "kb")
    except ImportError:
        pass
