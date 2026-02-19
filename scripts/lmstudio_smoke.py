from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True, help="Embedding model name loaded in LM Studio")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    try:
        r = requests.get(base + "/v1/models", timeout=10)
        r.raise_for_status()
        models = r.json()
        print("/v1/models OK")
        print(json.dumps(models, indent=2)[:2000])
    except Exception as e:  # noqa: BLE001
        print(f"/v1/models FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = {"model": args.model, "input": ["hello", "discord style test"]}
        r = requests.post(base + "/v1/embeddings", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        vec0 = data["data"][0]["embedding"]
        print("/v1/embeddings OK")
        print(f"dim={len(vec0)}")
    except Exception as e:  # noqa: BLE001
        print(f"/v1/embeddings FAILED: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
