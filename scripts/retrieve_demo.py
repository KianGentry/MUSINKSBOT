from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musinksbot.config import load_config
from musinksbot.embedder import build_embedder
from musinksbot.logging_utils import configure_logging
from musinksbot.retrieval import StyleRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    cfg = load_config(args.config)

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

    retriever = StyleRetriever(
        faiss_index_path=cfg.index.faiss_index_path,
        meta_jsonl_path=cfg.index.meta_jsonl_path,
        embedder=embedder,
    )

    hits = retriever.retrieve_similar_messages_scored(args.text, k=args.k)
    for i, h in enumerate(hits, start=1):
        print(f"#{i} score={h.score:.4f} ts={h.timestamp} row={h.source_row}")
        print(h.content)
        print("-")


if __name__ == "__main__":
    main()
