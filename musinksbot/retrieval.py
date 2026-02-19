from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musinksbot.embedder import Embedder
from musinksbot.indexing import load_faiss_index


@dataclass(frozen=True)
class RetrievedMessage:
    score: float
    content: str
    timestamp: str | None = None
    source_row: int | None = None


class StyleRetriever:
    def __init__(
        self,
        *,
        faiss_index_path: str | Path,
        meta_jsonl_path: str | Path,
        embedder: Embedder,
    ) -> None:
        self.index = load_faiss_index(faiss_index_path)
        self.embedder = embedder
        self.meta = self._load_meta(meta_jsonl_path)

    @staticmethod
    def _load_meta(path: str | Path) -> list[dict]:
        path = Path(path)
        items: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def retrieve_similar_messages_scored(self, context_text: str, k: int = 5) -> list[RetrievedMessage]:
        query = self.embedder.embed_text(context_text).astype(np.float32, copy=False)
        query = query.reshape(1, -1)
        scores, indices = self.index.search(query, k)
        out: list[RetrievedMessage] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            out.append(
                RetrievedMessage(
                    score=float(score),
                    content=m.get("content", ""),
                    timestamp=m.get("timestamp"),
                    source_row=m.get("source_row"),
                )
            )
        return out

    def retrieve_similar_messages(self, context_text: str, k: int = 5) -> list[str]:
        """Return top-k retrieved historical messages (content only)."""
        return [m.content for m in self.retrieve_similar_messages_scored(context_text, k=k)]

