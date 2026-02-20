from __future__ import annotations

import csv
import json
import re
from pathlib import Path


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _parse_discord_id(raw: str) -> int:
    s = (raw or "").strip()
    if not s:
        return 0
    try:
        return int(s)
    except Exception:
        pass
    try:
        return int(float(s))
    except Exception:
        return 0


def _clean_url(url: str) -> str:
    u = (url or "").strip()
    # Common trailing punctuation from CSV/content formatting.
    return u.rstrip(")]}>,.\"'\"")


def _is_tenor_url(url: str) -> bool:
    u = (url or "").lower()
    return "tenor.com/" in u or "media.tenor.com/" in u


def wants_gif(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Keep it tight to avoid accidental triggers.
    return bool(re.search(r"\b(send|post|drop)?\s*(a\s*)?(tenor\s*)?gif(s)?\b", t)) or t in {
        "gif",
        "gifs",
        "tenor",
    }


def load_tenor_links_from_csv(
    *,
    csv_path: Path,
    target_author_id: int,
    cache_path: Path,
) -> list[str]:
    """Extract Tenor links from the Discord CSV export.

    Notes:
    - `clean_messages.jsonl` typically has URLs replaced/removed (replace_urls=true),
      so we scan `discord.csv` to recover Tenor links.
    - Filters to the target author id to match the style corpus.
    """

    try:
        if cache_path.exists() and csv_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and "links" in cached and "csv_mtime" in cached:
                if float(cached["csv_mtime"]) == csv_path.stat().st_mtime:
                    links = cached.get("links")
                    if isinstance(links, list) and all(isinstance(x, str) for x in links):
                        return [x for x in links if x]
    except Exception:
        pass

    links_set: set[str] = set()
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if not header:
            return []

        def idx(name: str) -> int:
            try:
                return header.index(name)
            except ValueError:
                return -1

        i_author = idx("AuthorID")
        i_content = idx("Content")
        i_attach = idx("Attachments")

        for row in reader:
            if not row:
                continue

            if i_author >= 0 and target_author_id:
                author_id = _parse_discord_id(row[i_author] if i_author < len(row) else "")
                if author_id != int(target_author_id):
                    continue

            content = row[i_content] if i_content >= 0 and i_content < len(row) else ""
            attachments = row[i_attach] if i_attach >= 0 and i_attach < len(row) else ""
            blob = f"{content} {attachments}".strip()
            if not blob:
                continue
            # Cheap prefilter.
            if "tenor" not in blob.lower():
                continue

            for m in URL_RE.finditer(blob):
                u = _clean_url(m.group(0))
                if u and _is_tenor_url(u):
                    links_set.add(u)

    links = sorted(links_set)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"csv_mtime": csv_path.stat().st_mtime, "links": links}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return links
