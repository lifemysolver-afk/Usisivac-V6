"""
╔══════════════════════════════════════════════════════════════════════╗
║  BrainMassIngest — Masovni ChromaDB Ingest                          ║
║  Usisivac V6 | Trinity Protocol                                     ║
║  Integrisano iz: Trinity_AIMO_Loptica_Final / brain_mass_ingest.py  ║
╚══════════════════════════════════════════════════════════════════════╝

Usisava sve .py, .md, .json, .yaml fajlove iz zadatog direktorijuma
u ChromaDB kolekciju "massive_brain".

Koristi se za inicijalni ingest celokupne baze znanja projekta.
"""

import os, logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BrainMassIngest:
    """
    Masovni ingest fajlova u ChromaDB.
    Podržava chunking, deduplication i batch upload.
    """

    SUPPORTED_EXTENSIONS = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".ipynb"}
    EXCLUDE_DIRS = {"node_modules", "__pycache__", ".cache", ".local", "venv",
                    "chroma_db", "db", ".git", "catboost_info"}
    MAX_FILE_SIZE = 300_000  # 300KB
    CHUNK_MIN_LEN = 30
    BATCH_SIZE = 100

    def __init__(self, collection_name: str = "massive_brain",
                 db_path: str = None):
        self.collection_name = collection_name
        self.db_path = db_path or "/home/ubuntu/Usisivac-V6/chroma_db"

        import chromadb
        from core.rag_engine import _ef
        self.client = chromadb.PersistentClient(path=self.db_path)
        ef = _ef()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef
        )

    def scan_directory(self, root_dir: str) -> tuple:
        """Skenira direktorijum i vraća (documents, metadatas, ids)."""
        documents, metadatas, ids = [], [], []
        root = Path(root_dir)
        file_idx = 0

        for file_path in root.rglob("*"):
            # Preskoči isključene direktorijume
            if any(ex in file_path.parts for ex in self.EXCLUDE_DIRS):
                continue
            if not file_path.is_file():
                continue
            if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
                continue
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content.strip()) < 20:
                    continue

                # Chunking po dvostrukim newline-ovima
                chunks = [c.strip() for c in content.split("\n\n")
                          if len(c.strip()) >= self.CHUNK_MIN_LEN]

                for j, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "source": str(file_path),
                        "file_type": file_path.suffix,
                        "is_agent": ".agent" in str(file_path),
                    })
                    ids.append(f"brain_{file_idx}_{j}")

                file_idx += 1
                if file_idx % 500 == 0:
                    logger.info(f"Scanned {file_idx} files...")

            except Exception as e:
                logger.debug(f"Skip {file_path}: {e}")

        return documents, metadatas, ids

    def ingest(self, root_dir: str) -> dict:
        """Glavni ingest. Vraća statistike."""
        logger.info(f"Starting BrainMassIngest from: {root_dir}")
        docs, metas, ids = self.scan_directory(root_dir)

        if not docs:
            return {"status": "EMPTY", "files": 0, "chunks": 0}

        before = self.collection.count()

        # Batch upload
        for i in range(0, len(docs), self.BATCH_SIZE):
            end = min(i + self.BATCH_SIZE, len(docs))
            self.collection.upsert(
                documents=docs[i:end],
                metadatas=metas[i:end],
                ids=ids[i:end]
            )

        after = self.collection.count()
        result = {
            "status": "OK",
            "chunks_added": after - before,
            "total_in_db": after,
            "root_dir": root_dir,
        }
        logger.info(f"BrainMassIngest complete: {result}")
        return result

    def query(self, query_text: str, n_results: int = 5) -> list:
        """Pretražuje massive_brain kolekciju."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count() or 1)
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"text": d, "source": m.get("source", "")}
                for d, m in zip(docs, metas)]
