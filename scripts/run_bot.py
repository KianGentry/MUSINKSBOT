from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_bot.py` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _require_dependencies() -> None:
    try:
        import yaml  # noqa: F401
    except ModuleNotFoundError as e:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        msg = (
            "Missing dependency: PyYAML (import name 'yaml').\n\n"
            "You are probably running the bot with a Python interpreter that does not have the project's dependencies installed.\n\n"
            "Fix (recommended):\n"
            f"  {venv_python} -m pip install -r requirements.txt\n"
            f"  {venv_python} scripts/run_bot.py --config config.yaml\n\n"
            "Alternative: install requirements into the interpreter you're using (e.g. `python3 -m pip install -r requirements.txt`)."
        )
        raise SystemExit(msg) from e

from musinksbot.config import load_config
from musinksbot.discord_bot import MusinksDiscordClient, resolve_discord_token
from musinksbot.logging_utils import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    _require_dependencies()

    configure_logging(args.log_level)
    cfg = load_config(args.config)
    token = resolve_discord_token(cfg)

    client = MusinksDiscordClient(cfg)
    client.run(token)


if __name__ == "__main__":
    main()
