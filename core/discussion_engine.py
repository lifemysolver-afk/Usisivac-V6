import chromadb
from chromadb.config import Settings
import os
import json
from datetime import datetime

class DiscussionEngine:
    def __init__(self, persist_directory="./db/discussion_db"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="discussions")
        self.log_path = "logs/discussion_log.jsonl"
        os.makedirs("logs", exist_ok=True)

    def save_discussion(self, topic, participants, transcript, verdict):
        timestamp = datetime.now().isoformat()
        discussion_id = f"disc_{int(datetime.now().timestamp())}"
        
        # Save to ChromaDB for semantic search
        self.collection.add(
            documents=[transcript],
            metadatas=[{"topic": topic, "verdict": verdict, "timestamp": timestamp}],
            ids=[discussion_id]
        )
        
        # Save to JSONL for audit
        entry = {
            "id": discussion_id,
            "timestamp": timestamp,
            "topic": topic,
            "participants": participants,
            "verdict": verdict,
            "transcript": transcript
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        return discussion_id

    def get_relevant_discussions(self, query, n_results=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
