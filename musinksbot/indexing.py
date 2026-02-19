from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexBuildInfo:
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    num_vectors: int


def build_faiss_ip_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def save_build_info(info: IndexBuildInfo, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info.__dict__, f, ensure_ascii=False, indent=2)
