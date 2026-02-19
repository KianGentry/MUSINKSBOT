from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    p = Path("data/clean_messages.jsonl")
    lines = p.read_text(encoding="utf-8").splitlines()
    step = max(1, len(lines) // 12000)

    texts: list[str] = []
    for line in lines[::step]:
        try:
            obj = json.loads(line)
            t = str(obj.get("content", "")).strip().lower()
            if t:
                texts.append(t)
        except Exception:
            continue

    all_txt = "\n".join(texts)
    words = re.findall(r"[a-z']+", all_txt)
    c = Counter(words)

    print("sample_msgs", len(texts))
    for w in [
        "sall",
        "heh",
        "lol",
        "lmao",
        "bird",
        "gf",
        "girlfriend",
        "cringe",
        "villain",
        "mate",
        "somet",
        "prob",
    ]:
        print(w, c[w])

    for ph in [
        "fuck you",
        "fuck u",
        "watch your tone",
        "looooo",
        "post about",
        "gonna post",
        "upload",
        "publish",
    ]:
        print(ph, all_txt.count(ph))


if __name__ == "__main__":
    main()
