from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musinksbot.config import load_config
from musinksbot.embedder import build_embedder
from musinksbot.indexing import IndexBuildInfo, build_faiss_ip_index, save_build_info, save_faiss_index
from musinksbot.logging_utils import configure_logging


logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_meta_jsonl(items: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings/index even if output files already exist.",
    )
    args = ap.parse_args()

    configure_logging(args.log_level)
    cfg = load_config(args.config)

    if (
        not args.force
        and cfg.index.faiss_index_path.exists()
        and cfg.index.meta_jsonl_path.exists()
        and cfg.index.build_info_json.exists()
    ):
        logger.info("Index files already exist; skipping (use --force to rebuild).")
        return

    items = load_jsonl(cfg.dataset.output_jsonl)
    texts = [it["content"] for it in items]

    model = cfg.embeddings.model
    if cfg.embeddings.provider.lower() == "lm_studio" and cfg.lm_studio.embeddings_model:
        model = cfg.lm_studio.embeddings_model

    embedder = build_embedder(
        provider=cfg.embeddings.provider,
        model=model,
        device=cfg.embeddings.device,
        batch_size=cfg.embeddings.batch_size,
        lm_studio_base_url=cfg.lm_studio.base_url,
    )

    logger.info("Embedding %d messages...", len(texts))
    embeddings = embedder.embed_texts(texts)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("Embedder returned unexpected shape")

    logger.info("Building FAISS index (dim=%d)...", embeddings.shape[1])
    index = build_faiss_ip_index(embeddings)
    save_faiss_index(index, cfg.index.faiss_index_path)

    save_meta_jsonl(items, cfg.index.meta_jsonl_path)
    save_build_info(
        IndexBuildInfo(
            embedding_provider=cfg.embeddings.provider,
            embedding_model=model,
            embedding_dim=int(embeddings.shape[1]),
            num_vectors=int(embeddings.shape[0]),
        ),
        cfg.index.build_info_json,
    )

    logger.info("Saved index to %s", cfg.index.faiss_index_path)


if __name__ == "__main__":
    main()
