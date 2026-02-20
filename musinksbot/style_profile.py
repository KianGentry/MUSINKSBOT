from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


WORD_RE = re.compile(r"[a-z']+", re.IGNORECASE)


DEFAULT_MARKERS = [
    "im",
    "dont",
    "cant",
    "cus",
    "tbf",
    "pmo",
    "js",
    "mate",
    "pal",
    "nah",
    "yeah",
    "cheers",
]


@dataclass(frozen=True)
class StyleProfile:
    messages: int
    avg_chars: int
    p50_chars: float
    p90_chars: float
    markers: list[str]
    marker_counts: dict[str, int]
    recent_examples: list[str]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def build_style_profile(
    *,
    clean_jsonl_path: str | Path,
    markers: list[str] | None = None,
    recent_examples: int = 8,
    recent_tail: int = 400,
) -> StyleProfile:
    clean_jsonl_path = Path(clean_jsonl_path)
    markers = markers or DEFAULT_MARKERS

    lengths: list[int] = []
    marker_counts = {m: 0 for m in markers}
    tail: list[str] = []

    with clean_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            content = str(obj.get("content", ""))
            lengths.append(len(content))
            tokens = set(_tokenize(content))
            for m in markers:
                if m in tokens:
                    marker_counts[m] += 1
            tail.append(content)
            if len(tail) > recent_tail:
                tail.pop(0)

    if not lengths:
        raise ValueError("No messages found for style profile")

    lengths_sorted = sorted(lengths)
    p50 = float(statistics.median(lengths_sorted))
    p90 = float(lengths_sorted[int(0.9 * len(lengths_sorted)) - 1])

    # Take evenly spaced examples from the most recent tail to keep it deterministic.
    recent: list[str] = []
    if tail:
        step = max(1, len(tail) // max(1, recent_examples))
        for i in range(0, len(tail), step):
            recent.append(tail[i])
            if len(recent) >= recent_examples:
                break

    # Order markers by usage.
    markers_sorted = sorted(markers, key=lambda m: marker_counts.get(m, 0), reverse=True)

    return StyleProfile(
        messages=len(lengths),
        avg_chars=float(statistics.mean(lengths)),
        p50_chars=p50,
        p90_chars=p90,
        markers=markers_sorted,
        marker_counts=marker_counts,
        recent_examples=recent,
    )


def save_style_profile(profile: StyleProfile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, ensure_ascii=False, indent=2)


def load_style_profile(path: str | Path) -> StyleProfile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return StyleProfile(**raw)
