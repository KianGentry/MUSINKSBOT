from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musinksbot.config import load_config
from musinksbot.logging_utils import configure_logging
from musinksbot.preprocess import preprocess_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    cfg = load_config(args.config)

    preprocess_csv(
        csv_path=cfg.dataset.csv_path,
        output_jsonl=cfg.dataset.output_jsonl,
        stats_json=cfg.dataset.stats_json,
        target_author_id=cfg.target.author_id,
        min_chars=cfg.preprocess.min_chars,
        replace_urls=cfg.preprocess.replace_urls,
    )


if __name__ == "__main__":
    main()
