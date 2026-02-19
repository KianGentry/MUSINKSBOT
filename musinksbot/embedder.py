from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import requests


logger = logging.getLogger(__name__)


class Embedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.embed_texts([text])
        return vec[0]


@dataclass(frozen=True)
class SentenceTransformersEmbedder(Embedder):
    model_name: str
    device: str = "cpu"
    batch_size: int = 128

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        object.__setattr__(self, "_model", SentenceTransformer(self.model_name, device=self.device))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= 512,
        )
        return emb.astype(np.float32, copy=False)


@dataclass(frozen=True)
class LMStudioEmbedder(Embedder):
    base_url: str
    model: str
    timeout_s: float = 60.0
    max_batch: int = 128
    retries: int = 3
    backoff_s: float = 1.0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        url = self.base_url.rstrip("/") + "/v1/embeddings"
        all_vecs: list[list[float]] = []

        for start in range(0, len(texts), self.max_batch):
            batch = texts[start : start + self.max_batch]
            payload = {"model": self.model, "input": batch}

            last_err: Exception | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    resp = requests.post(url, json=payload, timeout=self.timeout_s)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("data", [])
                    # OpenAI-compatible embeddings API returns list with index ordering.
                    items_sorted = sorted(items, key=lambda x: x.get("index", 0))
                    for item in items_sorted:
                        all_vecs.append(item["embedding"])
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    sleep = self.backoff_s * attempt
                    logger.warning("Embeddings request failed (attempt %d/%d): %s", attempt, self.retries, e)
                    time.sleep(sleep)

            if last_err is not None:
                raise last_err

        arr = np.asarray(all_vecs, dtype=np.float32)
        # Normalize to unit length for cosine similarity in IndexFlatIP.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr


def build_embedder(
    *,
    provider: str,
    model: str,
    device: str = "cpu",
    batch_size: int = 128,
    lm_studio_base_url: str = "http://127.0.0.1:1234",
) -> Embedder:
    provider = provider.strip().lower()
    if provider == "sentence_transformers":
        return SentenceTransformersEmbedder(model_name=model, device=device, batch_size=batch_size)
    if provider == "lm_studio":
        return LMStudioEmbedder(base_url=lm_studio_base_url, model=model, max_batch=batch_size)
    raise ValueError(f"Unknown embeddings provider: {provider}")
