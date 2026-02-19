from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musinksbot.config import load_config
from musinksbot.logging_utils import configure_logging
from musinksbot.style_profile import build_style_profile, save_style_profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    cfg = load_config(args.config)

    profile = build_style_profile(clean_jsonl_path=cfg.dataset.output_jsonl)
    save_style_profile(profile, cfg.dataset.style_profile_json)
    print(f"wrote style profile to {cfg.dataset.style_profile_json}")


if __name__ == "__main__":
    main()
