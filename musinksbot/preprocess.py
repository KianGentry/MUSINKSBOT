from __future__ import annotations

import csv
import json
import logging
import re
import string
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
DISCORD_CUSTOM_EMOJI_RE = re.compile(r"<a?:[A-Za-z0-9_]+:\d+>")
DISCORD_COLON_EMOJI_RE = re.compile(r"(?::[A-Za-z0-9_~]+:)+")


@dataclass(frozen=True)
class PreprocessStats:
    rows_total: int = 0
    rows_target_author: int = 0
    kept: int = 0
    dropped_too_short: int = 0
    dropped_url_only: int = 0
    dropped_attachment_only: int = 0
    dropped_emoji_or_punct_only: int = 0
    dropped_empty: int = 0
    dropped_other: int = 0


def normalize_author_id(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""

    try:
        if "e" in s.lower():
            return str(int(Decimal(s)))
        if s.endswith(".0") and s.replace(".", "").isdigit():
            return s[:-2]
    except (InvalidOperation, ValueError):
        pass
    return s


def _attachments_present(attachments_field: str) -> bool:
    if attachments_field is None:
        return False
    return bool(str(attachments_field).strip())


def _is_url_only(text: str) -> bool:
    if not text.strip():
        return False
    stripped = URL_RE.sub("", text).strip()
    return stripped == ""


def _strip_discord_emoji_tokens(text: str) -> str:
    text = DISCORD_CUSTOM_EMOJI_RE.sub(" ", text)
    text = DISCORD_COLON_EMOJI_RE.sub(" ", text)
    return text


def _is_emoji_or_punct_only(text: str) -> bool:
    t = text.strip()
    if not t:
        return True

    # If it contains any letters/digits, keep.
    if any(ch.isalnum() for ch in t):
        return False

    # Remove known discord emoji tokens; then re-check.
    t2 = _strip_discord_emoji_tokens(t).strip()
    if not t2:
        return True

    # If after removing whitespace, everything is punctuation/symbol/mark, drop.
    for ch in t2:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat[0] in {"L", "N"}:  # letters/numbers
            return False
        if ch in string.punctuation:
            continue
        if cat[0] in {"P", "S", "M"}:  # punctuation/symbol/mark
            continue
        # Other categories (like separators) are fine to ignore.
    return True


def clean_message(
    content: str,
    attachments: str,
    *,
    min_chars: int,
    replace_urls: bool,
) -> str | None:
    content = "" if content is None else str(content)
    attachments = "" if attachments is None else str(attachments)
    raw = content.strip()

    if not raw:
        return None

    if _is_url_only(raw):
        return None

    # Replace URLs inside otherwise-meaningful messages to reduce embedding noise.
    if replace_urls:
        raw = URL_RE.sub("<url>", raw)

    # Collapse whitespace
    raw = re.sub(r"\s+", " ", raw).strip()

    if len(raw) < min_chars:
        return None

    if _attachments_present(attachments) and raw == "":
        return None

    if _is_emoji_or_punct_only(raw):
        return None

    return raw


def preprocess_csv(
    *,
    csv_path: str | Path,
    output_jsonl: str | Path,
    stats_json: str | Path,
    target_author_id: str,
    min_chars: int = 5,
    replace_urls: bool = True,
    log_every: int = 50_000,
) -> PreprocessStats:
    csv_path = Path(csv_path)
    output_jsonl = Path(output_jsonl)
    stats_json = Path(stats_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    target_author_id_norm = normalize_author_id(target_author_id)
    stats = PreprocessStats()

    logger.info("Preprocessing %s", csv_path)
    logger.info("Target AuthorID: %s", target_author_id_norm)

    rows_total = 0
    rows_target = 0
    kept = 0

    dropped_too_short = 0
    dropped_url_only = 0
    dropped_attachment_only = 0
    dropped_emoji_or_punct_only = 0
    dropped_empty = 0
    dropped_other = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f_in, output_jsonl.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        expected = {"AuthorID", "Author", "Date", "Content", "Attachments", "Reactions"}
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing = expected.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing expected columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            rows_total += 1

            author_id = normalize_author_id(row.get("AuthorID", ""))
            if author_id != target_author_id_norm:
                if rows_total % log_every == 0:
                    logger.info("Processed %d rows (kept=%d)", rows_total, kept)
                continue

            rows_target += 1
            content = row.get("Content", "")
            attachments = row.get("Attachments", "")
            date = row.get("Date", "")
            author = row.get("Author", "")

            raw = "" if content is None else str(content).strip()
            if not raw:
                if _attachments_present(str(attachments or "")):
                    dropped_attachment_only += 1
                else:
                    dropped_empty += 1
                continue

            if _is_url_only(raw):
                dropped_url_only += 1
                continue

            cleaned = clean_message(
                content=str(content or ""),
                attachments=str(attachments or ""),
                min_chars=min_chars,
                replace_urls=replace_urls,
            )
            if cleaned is None:
                if len(raw) < min_chars:
                    dropped_too_short += 1
                elif _is_emoji_or_punct_only(raw):
                    dropped_emoji_or_punct_only += 1
                else:
                    dropped_other += 1
                continue

            record = {
                "source_row": row_number,
                "author_id": author_id,
                "author": author,
                "timestamp": date,
                "content": cleaned,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if rows_total % log_every == 0:
                logger.info(
                    "Processed %d rows (target=%d kept=%d)",
                    rows_total,
                    rows_target,
                    kept,
                )

    stats = PreprocessStats(
        rows_total=rows_total,
        rows_target_author=rows_target,
        kept=kept,
        dropped_too_short=dropped_too_short,
        dropped_url_only=dropped_url_only,
        dropped_attachment_only=dropped_attachment_only,
        dropped_emoji_or_punct_only=dropped_emoji_or_punct_only,
        dropped_empty=dropped_empty,
        dropped_other=dropped_other,
    )

    with stats_json.open("w", encoding="utf-8") as f_stats:
        json.dump(asdict(stats), f_stats, ensure_ascii=False, indent=2)

    logger.info("Done. Kept %d messages", kept)
    return stats
