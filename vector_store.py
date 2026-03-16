"""
vector_store.py — in-memory vector store using numpy.
Embeds documents with OpenAI text-embedding-3-small and retrieves
the top-k most similar chunks via cosine similarity.

No external vector database required.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loader import Document


class VectorStore:
    """
    Stores document embeddings in memory (numpy array).
    Can be persisted to / loaded from disk as a .pkl file.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._docs: list["Document"] = []
        self._matrix = None   # np.ndarray of shape (N, D) once built

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)

    def _embed_texts(self, texts: list[str]) -> "list[list[float]]":
        client = self._get_client()
        # OpenAI supports batches of up to 2048 inputs
        BATCH = 512
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            response = client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    # ------------------------------------------------------------------
    # Build / index
    # ------------------------------------------------------------------

    def build(self, docs: list["Document"], show_progress: bool = True) -> None:
        import numpy as np

        if show_progress:
            print(f"  Embedding {len(docs)} chunks with {self.model} ...", flush=True)

        texts = [d.text for d in docs]
        embeddings = self._embed_texts(texts)

        self._docs = docs
        self._matrix = np.array(embeddings, dtype="float32")
        # L2-normalise for fast cosine sim via dot product
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        self._matrix /= np.where(norms == 0, 1, norms)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> list[tuple["Document", float]]:
        import numpy as np

        if self._matrix is None:
            raise RuntimeError("Vector store is empty. Call build() first.")

        emb = self._embed_texts([question])[0]
        q = np.array(emb, dtype="float32")
        q /= np.linalg.norm(q) or 1.0

        scores = self._matrix @ q
        top_indices = scores.argsort()[::-1][:top_k]

        return [(self._docs[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"docs": self._docs, "matrix": self._matrix, "model": self.model}, f)
        print(f"  Index saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(model=data["model"])
        store._docs = data["docs"]
        store._matrix = data["matrix"]
        print(f"  Loaded {len(store._docs)} chunks from {path}")
        return store

    def __len__(self) -> int:
        return len(self._docs)
