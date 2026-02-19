from __future__ import annotations

import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import discord
from discord.ext import tasks

from musinksbot.config import AppConfig
from musinksbot.embedder import build_embedder
from musinksbot.lmstudio_client import LMStudioChatConfig, LMStudioClient
from musinksbot.prompting import build_prompt, extract_direct_request
from musinksbot.retrieval import StyleRetriever
from musinksbot.style_profile import StyleProfile, load_style_profile


logger = logging.getLogger(__name__)


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z']+")
QUESTION_WORD_RE = re.compile(r"\b(why|what|how|when|where|who)\b", re.IGNORECASE)

EH_WORD_RE = re.compile(r"\beh+\b", re.IGNORECASE)

FILLER_WORDS = {
    "eh",
    "ehh",
    "uh",
    "um",
    "erm",
    "idk",
}

CONTENT_IGNORE_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "bc",
    "because",
    "but",
    "cos",
    "cus",
    "cuz",
    "did",
    "do",
    "dont",
    "for",
    "from",
    "get",
    "got",
    "how",
    "i",
    "idk",
    "im",
    "in",
    "is",
    "it",
    "its",
    "like",
    "mate",
    "me",
    "nah",
    "no",
    "not",
    "of",
    "ok",
    "on",
    "or",
    "same",
    "so",
    "tbf",
    "that",
    "the",
    "then",
    "to",
    "u",
    "ur",
    "was",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "yeah",
    "ye",
    "yep",
    "yup",
    "you",
    "your",
}

POTTY_HUMOUR_RE = re.compile(
    r"\b("
    r"poo(p)?\w*|"
    r"shit\w*|shite\w*|"
    r"fart\w*|"
    r"bum\w*|arse\w*|ass\w*|"
    r"piss\w*|pee\w*|"
    r"wee\w*|"
    r"wank\w*|"
    r"dick\w*|cock\w*|"
    r"balls\w*|bollocks\w*|"
    r"penis\w*|"
    r"boob\w*|tits\w*|"
    r"cum\w*"
    r")\b",
    re.IGNORECASE,
)


PUNCT_ALWAYS_STRIP_RE = re.compile(r"[!\"\“\”\(\)\[\]\{\};:]")


def _is_potty_humour(text: str) -> bool:
    return bool(POTTY_HUMOUR_RE.search((text or "").lower()))


def _last_user_text(recent: list[tuple[str, str]]) -> str:
    for author, content in reversed(recent):
        if author.strip().lower() == "musinks":
            continue
        c = (content or "").strip()
        if c:
            return c
    return ""


def _custom_emoji_by_name(guild: discord.Guild | None, name: str) -> discord.Emoji | None:
    if guild is None:
        return None
    for e in guild.emojis:
        if (e.name or "").lower() == name.lower():
            return e
    return None


def _pick_funny_reaction(guild: discord.Guild | None) -> str | discord.Emoji:
    # Prefer server custom emojis if present.
    for n in ("ICANT", "nevo1sICANT"):
        e = _custom_emoji_by_name(guild, n)
        if e is not None:
            return e
    # Fallback to a common equivalent.
    return "😭"


def _parse_discord_id(raw: str) -> int:
    """Parse config target.author_id which may be scientific notation."""
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


def _is_live_style_candidate(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if len(t) < 4 or len(t) > 80:
        return False
    if "http://" in t or "https://" in t:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    # keep mostly short chat lines (style-y)
    return True


def _apply_punctuation_policy_preserving_urls(text: str) -> str:
    """Apply punctuation rules to match the user's style.

    Rules:
    - Single sentence: no punctuation (keep commas only if already used for lists; keep ? if it's a question)
    - Multiple sentences: allow separators like . ? ,
    - Always strip quotes/brackets/colons/semicolons/exclamation marks
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # Punctuation rules here are strict; we rely on multi-message splitting instead.
    has_commas = "," in raw

    parts: list[str] = []
    last = 0
    for m in URL_RE.finditer(raw):
        chunk = raw[last : m.start()]
        chunk = re.sub(r"([a-z0-9])['’]([a-z0-9])", r"\1\2", chunk, flags=re.IGNORECASE)
        chunk = PUNCT_ALWAYS_STRIP_RE.sub("", chunk)
        # Always remove sentence punctuation; keep commas only if they were used (lists).
        chunk = chunk.replace(".", "").replace("?", "")
        if not has_commas:
            chunk = chunk.replace(",", "")
        chunk = re.sub(r"\s+", " ", chunk)
        parts.append(chunk)
        parts.append(raw[m.start() : m.end()])
        last = m.end()

    chunk = raw[last:]
    chunk = re.sub(r"([a-z0-9])['’]([a-z0-9])", r"\1\2", chunk, flags=re.IGNORECASE)
    chunk = PUNCT_ALWAYS_STRIP_RE.sub("", chunk)
    chunk = chunk.replace(".", "").replace("?", "")
    if not has_commas:
        chunk = chunk.replace(",", "")
    chunk = re.sub(r"\s+", " ", chunk)
    parts.append(chunk)

    out = "".join(parts).strip()
    return out


def _lowercase_preserving_urls(text: str) -> str:
    parts: list[str] = []
    last = 0
    for m in URL_RE.finditer(text):
        parts.append(text[last : m.start()].lower())
        parts.append(text[m.start() : m.end()])
        last = m.end()
    parts.append(text[last:].lower())
    return "".join(parts)


def postprocess_bot_text(
    text: str,
    *,
    last_user_text: str = "",
    max_chars: int | None = None,
) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Remove a couple of common wrappers.
    for prefix in ("assistant:", "musinks:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix) :].strip()
    # Enforce lowercase (user requirement) while keeping urls intact.
    t = _lowercase_preserving_urls(t)
    # Guardrail: avoid out-of-corpus token that the model tends to overuse.
    t = re.sub(r"\bwot\b", "what", t)
    # Align to user's shorthand (based on corpus).
    t = re.sub(r"\bsomethin\b", "somet", t)
    t = re.sub(r"\bprobs\b", "prob", t)
    t = re.sub(r"\bexp\b", "experienced", t)
    # Avoid invented contraction not present in corpus.
    t = re.sub(r"\bsall\b", "its all", t)
    # Reduce reflexive "heh" openers.
    t = re.sub(r"^heh+\s+", "", t)

    # Avoid overusing "eh" unless the user already used it.
    if not EH_WORD_RE.search((last_user_text or "").lower()):
        t = EH_WORD_RE.sub("", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
    # Collapse lolol / lololol / lolololol etc.
    t = re.sub(r"\b(?:lo){2,}l\b", "lol", t)
    # Cap lol/lmao repetition.
    t = re.sub(r"\b(lol|lmao)(\s+\1)+\b", r"\1", t)
    # Avoid out-of-style americanisms.
    t = re.sub(r"\bdude\b", "mate", t)

    # Avoid leaning on "meme" as a filler word.
    t = re.sub(r"\bmemes?\b", "joke", t)

    funny = _is_potty_humour(last_user_text)
    # Strongly suppress lol/lmao in general.
    if funny:
        # If laughing, prefer a single lmaooo (rare).
        t = re.sub(r"\b(lol|lmao)\b", "lmaooo", t)
    else:
        t = re.sub(r"\b(lol|lmao)\b", "", t)
        t = re.sub(r"\s{2,}", " ", t).strip()

    # If it still contains a laugh token, unify and optionally uppercase it.
    t = re.sub(r"\blma+o+\b", "LMAOOO", t)

    # Apply punctuation rules.
    t = _apply_punctuation_policy_preserving_urls(t)

    # Drop standalone signature lines the model sometimes emits.
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln and ln not in {"musinks", "musinks app"}]
    t = "\n".join(lines).strip()

    if max_chars is not None and max_chars > 0 and len(t) > int(max_chars):
        cut = t[: int(max_chars)].rstrip()
        # Avoid cutting mid-word.
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0].rstrip()
        t = cut
    return t.strip()


def relevance_issues(*, reply: str, last_user_text: str) -> list[str]:
    """Detect replies that ignore what the user just said.

    We keep this simple: if the user said specific content words, the reply should
    contain at least one of them or ask a clarifying question.
    """
    r = (reply or "").strip().lower()
    u = (last_user_text or "").strip().lower()
    if not r or not u:
        return []

    user_cw = _content_words(u)
    if len(user_cw) >= 2:
        reply_cw = _content_words(r)
        # Allow short clarification questions.
        if _looks_like_question(r):
            return []
        if len(user_cw & reply_cw) == 0:
            return ["low_relevance"]

    # Avoid spitting out links/gifs unless the user already posted a link.
    if URL_RE.search(r) and not URL_RE.search(u):
        return ["unsolicited_link"]

    return []


def _split_for_sending(raw_text: str, *, max_parts: int = 2) -> list[str]:
    t = (raw_text or "").strip()
    if not t:
        return []
    # Prefer explicit newline separation (model can do this).
    if "\n" in t:
        parts = [p.strip() for p in re.split(r"\n+", t) if p.strip()]
    else:
        parts = [p.strip() for p in re.split(r"[\.\?!]+", t) if p.strip()]
    if not parts:
        return []
    # If it looks like one sentence, keep it as one.
    if len(parts) == 1:
        return parts
    return parts[: max(1, int(max_parts))]


def _words(s: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")]


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Punctuation is often stripped; treat question-words as a question signal.
    return bool(QUESTION_WORD_RE.search(t))


def _similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _content_words(s: str) -> set[str]:
    toks = [w.lower() for w in WORD_RE.findall(s or "")]
    return {w for w in toks if w not in CONTENT_IGNORE_WORDS and len(w) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        return []
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(0, len(words) - n + 1)]


def phrase_repetition_issues(
    *,
    reply: str,
    recent_bot_texts: list[str],
    min_n: int = 3,
    max_n: int = 5,
    max_prev: int = 14,
) -> list[str]:
    """Detect reused short phrases from recent bot messages.

    This is a stronger signal than similarity for the "buzzword / catchphrase" failure mode.
    """
    r_words = [w.lower() for w in WORD_RE.findall((reply or "").lower())]
    if len(r_words) < min_n:
        return []

    prev_phrases: set[tuple[str, ...]] = set()
    for prev in recent_bot_texts[-int(max_prev) :]:
        p_words = [w.lower() for w in WORD_RE.findall((prev or "").lower())]
        if not p_words:
            continue
        for n in range(int(min_n), int(max_n) + 1):
            for ng in _ngrams(p_words, n):
                # Skip phrases that are entirely stopwords.
                if all(w in CONTENT_IGNORE_WORDS for w in ng):
                    continue
                prev_phrases.add(ng)

    hits: list[str] = []
    for n in range(int(min_n), int(max_n) + 1):
        for ng in _ngrams(r_words, n):
            if ng in prev_phrases:
                hits.append(" ".join(ng))

    # de-dupe, keep first-seen order
    return list(dict.fromkeys(hits))


def repetition_issues(*, reply: str, recent_bot_texts: list[str], threshold: float = 0.88) -> list[str]:
    """Return a list of recent bot messages that are too similar to the draft reply."""
    reply = (reply or "").strip()
    if not reply:
        return []
    hits: list[str] = []
    reply_cw = _content_words(reply)
    for prev in recent_bot_texts:
        prev = (prev or "").strip()
        if not prev:
            continue
        if reply == prev:
            hits.append(prev)
            continue
        if _similarity(reply, prev) >= float(threshold):
            hits.append(prev)
            continue

        # For short replies, also catch near-paraphrases via content-word overlap.
        if len(reply) <= 80:
            prev_cw = _content_words(prev)
            if len(reply_cw) >= 3 and len(prev_cw) >= 3 and _jaccard(reply_cw, prev_cw) >= 0.85:
                hits.append(prev)
    return hits


def nonsense_issues(*, reply: str) -> list[str]:
    t = (reply or "").strip().lower()
    if not t:
        return ["empty"]

    toks = _words(t)
    if not toks:
        return ["no_words"]

    # e.g. "mate mate mate" / "idk idk idk"
    for i in range(len(toks) - 2):
        if toks[i] == toks[i + 1] == toks[i + 2]:
            return ["triple_repeat"]

    # Very low diversity often reads as looping/garbled.
    if len(toks) >= 8:
        uniq = len(set(toks))
        if uniq / len(toks) < 0.45:
            return ["low_diversity"]

    # Too filler-heavy.
    if len(toks) >= 5:
        filler = sum(1 for w in toks if w in FILLER_WORDS)
        if filler / len(toks) >= 0.34:
            return ["filler_heavy"]

    return []


def grounding_issues(*, reply: str, context_text: str, allow_unknown_long_words: int = 0) -> list[str]:
    """Return a list of likely off-context topic words in the reply.

    Heuristic: flag long-ish words that do not appear anywhere in the recent context,
    excluding common fillers.
    """
    context_words = set(_words(context_text))

    # Common function words / fillers / style markers that are OK to introduce.
    allowed = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "bc",
        "because",
        "but",
        "cant",
        "cos",
        "cus",
        "cuz",
        "did",
        "do",
        "dont",
        "fair",
        "for",
        "from",
        "get",
        "got",
        "how",
        "i",
        "idk",
        "im",
        "in",
        "is",
        "it",
        "its",
        "js",
        "like",
        "lmao",
        "lol",
        "mate",
        "me",
        "nah",
        "no",
        "not",
        "of",
        "ok",
        "on",
        "or",
        "pal",
        "pmo",
        "same",
        "so",
        "tbf",
        "that",
        "the",
        "then",
        "tho",
        "though",
        "this",
        "to",
        "u",
        "ur",
        "very",
        "really",
        "maybe",
        "kinda",
        "sorta",
        "anyway",
        "anyways",
        "what",
        "when",
        "where",
        "who",
        "why",
        "yeah",
        "yea",
        "yep",
        "you",
        "your",
    }

    bad: list[str] = []
    for w in _words(reply):
        if len(w) < 6:
            continue
        if w in allowed:
            continue
        if w in context_words:
            continue
        bad.append(w)

    # Prevent common "made-up plans" drift (these are short words so they won't be caught above).
    context_words = set(_words(context_text))
    plan_words = {"post", "posting", "upload", "publish", "release", "announce"}
    for w in _words(reply):
        if w in plan_words and w not in context_words:
            bad.append(w)

    # Allow a small amount of novelty; beyond that, treat as likely off-topic.
    if len(set(bad)) <= allow_unknown_long_words:
        return []
    # Return a de-duplicated list in first-seen order.
    seen: set[str] = set()
    out: list[str] = []
    for w in bad:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def score_engagement_text(
    *,
    content: str,
    age_seconds: float,
    is_direct_reply: bool,
    mentions_bot: bool,
    bot_name: str,
    bot_asked_question: bool,
    last_bot_text: str = "",
) -> float:
    """Heuristic score for whether a message is engaging with the bot.

    Inputs are plain values so this can be tested/tuned without Discord objects.
    """
    score = 0.0
    content = (content or "").strip().lower()
    if not content:
        return 0.0

    if is_direct_reply:
        score += 5.0

    if mentions_bot:
        score += 5.0

    # If the bot just asked a question, most immediate follow-ups are relevant.
    if bot_asked_question and len(content) >= 2:
        score += 2.0

    bot_name = (bot_name or "").strip().lower()
    if bot_name and bot_name in content:
        score += 3.0
    if "musinks" in content:
        score += 2.0

    if "?" in content:
        score += 1.0
    # Question words even without a question mark.
    if re.search(r"\b(why|what|how|when|where|who)\b", content):
        score += 0.75
    if re.search(r"\b(you|your|u|ur)\b", content):
        score += 1.0
    if re.search(r"\b(thanks|cheers)\b", content):
        score += 0.5

    # If the user uses words from the bot's last message, it's likely a follow-up.
    if last_bot_text:
        stop = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "bc",
            "because",
            "but",
            "can",
            "cant",
            "cos",
            "cus",
            "cuz",
            "did",
            "do",
            "dont",
            "for",
            "from",
            "get",
            "got",
            "how",
            "i",
            "idk",
            "im",
            "in",
            "is",
            "it",
            "its",
            "js",
            "like",
            "me",
            "my",
            "nah",
            "no",
            "not",
            "of",
            "ok",
            "on",
            "or",
            "so",
            "tbf",
            "that",
            "the",
            "then",
            "tho",
            "this",
            "to",
            "u",
            "ur",
            "what",
            "when",
            "where",
            "who",
            "why",
            "yeah",
            "you",
            "your",
            "musinks",
        }

        def kw(s: str) -> set[str]:
            return {w for w in _words(s) if len(w) >= 4 and w not in stop}

        overlap = kw(content) & kw(last_bot_text)
        if overlap:
            score += 2.5
            if len(overlap) >= 2:
                score += 0.5

    # Short acknowledgements are weaker signals.
    if len(content) >= 12:
        score += 0.5

    # Very soon after the bot speaks, slightly boost.
    if age_seconds <= 120:
        score += 0.5

    return score


@dataclass
class EngagementState:
    until_monotonic: float = 0.0
    last_reply_monotonic: float = 0.0
    last_bot_message_id: int = 0
    last_bot_message_monotonic: float = 0.0
    last_bot_message_text: str = ""
    last_bot_asked_question: bool = False
    author_scores: dict[int, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.author_scores is None:
            self.author_scores = {}


class MusinksDiscordClient(discord.Client):
    def __init__(self, cfg: AppConfig) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True  # must be enabled in Discord developer portal

        super().__init__(intents=intents)

        self.cfg = cfg
        self.state = EngagementState()
        self.target_author_id = _parse_discord_id(cfg.target.author_id)

        self.live_style_tail: deque[str] = deque(maxlen=250)
        self.live_style_path = Path("data/live_style_tail.jsonl")
        try:
            if self.live_style_path.exists():
                with self.live_style_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        t = line.strip()
                        if _is_live_style_candidate(t):
                            self.live_style_tail.append(t)
        except Exception:
            pass

        self.style_profile: StyleProfile | None = None
        try:
            if cfg.dataset.style_profile_json.exists():
                self.style_profile = load_style_profile(cfg.dataset.style_profile_json)
                logger.info("Loaded style profile: %s", cfg.dataset.style_profile_json)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load style profile (%s): %s", cfg.dataset.style_profile_json, e)

        model = cfg.embeddings.model
        if cfg.embeddings.provider.lower() == "lm_studio" and cfg.lm_studio.embeddings_model:
            model = cfg.lm_studio.embeddings_model

        self.embedder = build_embedder(
            provider=cfg.embeddings.provider,
            model=model,
            device=cfg.embeddings.device,
            batch_size=cfg.embeddings.batch_size,
            lm_studio_base_url=cfg.lm_studio.base_url,
        )
        self.retriever = StyleRetriever(
            faiss_index_path=cfg.index.faiss_index_path,
            meta_jsonl_path=cfg.index.meta_jsonl_path,
            embedder=self.embedder,
        )
        self.llm = LMStudioClient(
            LMStudioChatConfig(
                base_url=cfg.lm_studio.base_url,
                model=cfg.generation.model,
                temperature=cfg.generation.temperature,
                max_tokens=cfg.generation.max_tokens,
                top_p=cfg.generation.top_p,
                stop=cfg.generation.stop,
                seed=cfg.generation.seed,
                retries=cfg.generation.retries,
            )
        )

        self._spontaneous_task.change_interval(minutes=cfg.discord.spontaneous_interval_minutes)

    async def setup_hook(self) -> None:
        self._spontaneous_task.start()

    def _in_engagement(self) -> bool:
        return time.monotonic() < self.state.until_monotonic

    def _cooldown_ok(self) -> bool:
        return (time.monotonic() - self.state.last_reply_monotonic) >= self.cfg.discord.cooldown_seconds

    async def _get_channel(self) -> discord.TextChannel:
        if not self.cfg.discord.channel_id:
            raise RuntimeError("discord.channel_id must be set in config.yaml")
        ch = self.get_channel(self.cfg.discord.channel_id)
        if ch is None:
            ch = await self.fetch_channel(self.cfg.discord.channel_id)
        if not isinstance(ch, discord.TextChannel):
            raise RuntimeError("Configured channel_id is not a text channel")
        return ch

    async def _fetch_recent_context(self, channel: discord.TextChannel) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for msg in channel.history(limit=int(self.cfg.discord.context_messages)):
            if msg.author is None:
                continue
            if msg.author.bot:
                # Include our own messages so the model can follow its own questions.
                if self.user is None or msg.author.id != self.user.id:
                    continue
                author = "musinks"
            else:
                author = msg.author.display_name
            content = (msg.content or "").strip()
            if not content:
                continue
            out.append((author, content))
        out.reverse()
        return out

    async def _generate_text(self, *, recent: list[tuple[str, str]], spontaneous: bool) -> str:
        last_user_text = _last_user_text(recent)
        # Retrieve examples based on style markers rather than conversation semantics.
        # This reduces topic bleed (examples are for wording/tone, not content).
        if self.style_profile is not None and self.style_profile.markers:
            style_query = " ".join([m for m in self.style_profile.markers[:12] if m])
        else:
            style_query = "im mate dont yeah cus cant js nah pal cheers tbf pmo"

        retrieval_k = int(self.cfg.discord.retrieval_k)
        # Only use the live style tail for spontaneous posts.
        # For replies it tends to cause catchphrase parroting / "buzzwords".
        if spontaneous:
            live_examples = [t for t in list(self.live_style_tail)[-30:] if _is_live_style_candidate(t)]
            live_examples = live_examples[-6:]
        else:
            live_examples = []

        if (not spontaneous) or retrieval_k <= 0:
            # For replies, keep examples minimal to stay coherent/on-topic.
            style_examples = []
        else:
            style_examples = live_examples + self.retriever.retrieve_similar_messages(style_query, k=retrieval_k)
        memory_snippets: list[str] = []
        if not spontaneous:
            direct_request = extract_direct_request(recent)
            dr = (direct_request or "").lower()
            if any(k in dr for k in ("nerdy", "program", "coding", "code", "os", "kernel", "compiler", "asm", "assembler")):
                # Pull a couple of relevant past lines to ground the nerdy response.
                memory_snippets = self.retriever.retrieve_similar_messages(
                    "custom os kernel programming compiler assembler",
                    k=3,
                )

        prompt = build_prompt(
            recent_messages=recent,
            style_examples=style_examples,
            memory_snippets=memory_snippets,
            style_profile=self.style_profile,
            spontaneous=spontaneous,
        )

        raw_text = self.llm.generate_response(system=prompt.system, user=prompt.user)

        # Enforce shorter replies based on observed style distribution.
        if spontaneous:
            max_chars = 140
        elif self.style_profile is not None:
            max_chars = max(35, int(self.style_profile.p50_chars) + 25)
        else:
            max_chars = 70

        # Split into up to 2 short messages if the model produced multiple sentences.
        parts_raw = _split_for_sending(raw_text, max_parts=2)
        if len(parts_raw) <= 1:
            text = postprocess_bot_text(raw_text, last_user_text=last_user_text, max_chars=max_chars)
        else:
            # Process each part as a single short message.
            processed = [
                postprocess_bot_text(p, last_user_text=last_user_text, max_chars=max_chars)
                for p in parts_raw
            ]
            processed = [p for p in processed if p]
            text = "\n".join(processed[:2])

        # Collect recent bot messages for repetition checks.
        recent_bot_texts = [c for a, c in recent[-20:] if a == "musinks"]

        # Guardrail: if the model drifts into random topics, force a single rewrite
        # that stays grounded in the chat context.
        context_text = "\n".join(f"{a}: {c}" for a, c in recent[-12:])
        if memory_snippets:
            context_text = context_text + "\n" + "\n".join(memory_snippets)
        # For grounding, check each part.
        parts_for_checks = [p.strip() for p in text.split("\n") if p.strip()]
        bad_words: list[str] = []
        for p in parts_for_checks:
            bad_words.extend(grounding_issues(reply=p, context_text=context_text))
        rep_hits = repetition_issues(reply=text, recent_bot_texts=recent_bot_texts)
        # Also check each part; short replies repeat more often.
        for p in parts_for_checks:
            rep_hits.extend(
                repetition_issues(
                    reply=p,
                    recent_bot_texts=recent_bot_texts,
                    threshold=0.84 if len(p) <= 60 else 0.88,
                )
            )
        # de-dupe while preserving order
        rep_hits = list(dict.fromkeys([h for h in rep_hits if h]))

        phrase_hits: list[str] = []
        phrase_hits.extend(phrase_repetition_issues(reply=text, recent_bot_texts=recent_bot_texts))
        for p in parts_for_checks:
            phrase_hits.extend(phrase_repetition_issues(reply=p, recent_bot_texts=recent_bot_texts))
        phrase_hits = list(dict.fromkeys([h for h in phrase_hits if h]))

        nonsense_hits: list[str] = []
        nonsense_hits.extend(nonsense_issues(reply=text))
        for p in parts_for_checks:
            nonsense_hits.extend(nonsense_issues(reply=p))
        nonsense_hits = list(dict.fromkeys([h for h in nonsense_hits if h]))

        relevance_hits: list[str] = []
        for p in parts_for_checks:
            relevance_hits.extend(relevance_issues(reply=p, last_user_text=last_user_text))
        relevance_hits = list(dict.fromkeys([h for h in relevance_hits if h]))

        if bad_words or rep_hits or nonsense_hits or relevance_hits or phrase_hits:
            problems: list[str] = []
            if bad_words:
                problems.append("unrelated topic words: " + ", ".join(bad_words[:12]))
            if rep_hits:
                problems.append("too similar to previous bot messages")
            if phrase_hits:
                problems.append("reused catchphrases: " + ", ".join(phrase_hits[:6]))
            if nonsense_hits:
                problems.append("nonsensical/looping output")
            if relevance_hits:
                problems.append("not answering the last user message")

            rewrite_user = (
                prompt.user
                + "\n\nrewrite:\n"
                + "- issues: "
                + "; ".join(problems)
                + "\n- rewrite the message staying strictly on the current chat topic\n"
                + "- answer the users last message directly\n"
                + "- do not repeat the same phrasing as earlier bot messages\n"
                + ("- avoid these exact phrases: " + ", ".join(phrase_hits[:8]) + "\n" if phrase_hits else "")
                + "- if you listed unrelated topic words above, do not mention them\n"
                + "- do not say eh (unless the user said it first)\n"
                + "- keep it short and all lowercase\n"
                + "- it must be coherent and make sense\n"
            )
            raw2 = self.llm.generate_response(system=prompt.system, user=rewrite_user)
            parts2 = _split_for_sending(raw2, max_parts=2)
            if len(parts2) <= 1:
                text2 = postprocess_bot_text(raw2, last_user_text=last_user_text, max_chars=max_chars)
            else:
                processed2 = [
                    postprocess_bot_text(p, last_user_text=last_user_text, max_chars=max_chars)
                    for p in parts2
                ]
                processed2 = [p for p in processed2 if p]
                text2 = "\n".join(processed2[:2])
            if text2:
                return text2

        return text

    async def _send_text_parts(
        self,
        *,
        channel: discord.TextChannel,
        parts_text: str,
        reply_to: discord.Message | None = None,
    ) -> discord.Message | None:
        parts = [p.strip() for p in (parts_text or "").split("\n") if p.strip()]
        if not parts:
            return None
        if len(parts) == 1:
            if reply_to is not None:
                return await reply_to.reply(parts[0], mention_author=False)
            return await channel.send(parts[0])

        # Two parts: send back-to-back. Reply with first if we have a message to reply to.
        first = parts[0]
        second = parts[1]
        sent1: discord.Message
        if reply_to is not None:
            sent1 = await reply_to.reply(first, mention_author=False)
        else:
            sent1 = await channel.send(first)
        sent2 = await channel.send(second)
        return sent2

    def _enter_engagement(self) -> None:
        self.state.until_monotonic = time.monotonic() + (60.0 * float(self.cfg.discord.engagement_window_minutes))

    def _record_bot_message(self, msg: discord.Message | None) -> None:
        if msg is None:
            return
        self.state.last_bot_message_id = int(msg.id)
        self.state.last_bot_message_monotonic = time.monotonic()
        self.state.last_bot_message_text = (msg.content or "").strip()
        self.state.last_bot_asked_question = _looks_like_question(self.state.last_bot_message_text)
        self.state.author_scores.clear()

    def _score_engagement(self, message: discord.Message) -> float:
        content = (message.content or "").strip()
        if not content:
            return 0.0
        # Only consider messages after the bot has spoken recently.
        if not self.state.last_bot_message_id or int(message.id) <= int(self.state.last_bot_message_id):
            return 0.0

        age = time.monotonic() - float(self.state.last_bot_message_monotonic)
        if age > float(self.cfg.discord.engagement_max_age_seconds):
            return 0.0

        is_direct_reply = False
        if message.reference and message.reference.message_id:
            is_direct_reply = int(message.reference.message_id) == int(self.state.last_bot_message_id)

        mentions_bot = self.user is not None and self.user.mentioned_in(message)
        bot_name = self.user.name if self.user is not None else ""
        return score_engagement_text(
            content=content,
            age_seconds=age,
            is_direct_reply=is_direct_reply,
            mentions_bot=mentions_bot,
            bot_name=bot_name,
            bot_asked_question=bool(self.state.last_bot_asked_question),
            last_bot_text=str(self.state.last_bot_message_text or ""),
        )

    @tasks.loop(minutes=30)
    async def _spontaneous_task(self) -> None:
        if not self.is_ready():
            return
        try:
            channel = await self._get_channel()
            recent = await self._fetch_recent_context(channel)
            text = await self._generate_text(recent=recent, spontaneous=True)
            if text:
                sent = await self._send_text_parts(channel=channel, parts_text=text)
                self._record_bot_message(sent)
                self._enter_engagement()
                logger.info("Spontaneous message sent; engagement window started")
        except Exception as e:  # noqa: BLE001
            logger.exception("Spontaneous task failed: %s", e)

    async def on_message(self, message: discord.Message) -> None:
        if message.author is None:
            return
        if self.user is not None and message.author.id == self.user.id:
            return
        if self.cfg.discord.ignore_bots and message.author.bot:
            return
        if not isinstance(message.channel, discord.TextChannel):
            return
        if self.cfg.discord.channel_id and message.channel.id != self.cfg.discord.channel_id:
            return

        # Reactions can happen anytime (even outside engagement) for potty/schoolboy humour.
        try:
            content = (message.content or "").strip()
            if content and _is_potty_humour(content):
                # Keep it uncommon.
                if random.random() < 0.35:
                    await message.add_reaction(_pick_funny_reaction(message.guild))
        except Exception:
            # Missing perms / unknown emoji etc - ignore.
            pass

        # Lightweight local "training": keep a tail of the target author's short lines.
        try:
            if self.target_author_id and int(message.author.id) == int(self.target_author_id):
                content = (message.content or "").strip()
                if _is_live_style_candidate(content):
                    self.live_style_tail.append(content)
                    self.live_style_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.live_style_path.open("a", encoding="utf-8") as f:
                        f.write(content.replace("\n", " ").strip() + "\n")
        except Exception:
            pass

        mentioned = self.user is not None and self.user.mentioned_in(message)

        # Mentions can start a conversation even outside engagement.
        if mentioned and self.cfg.discord.reply_when_mentioned:
            if not self._in_engagement():
                self._enter_engagement()
                logger.info("Mention received; engagement window started")
            # Mention replies happen immediately (subject to cooldown).
            if not self._cooldown_ok():
                return
            try:
                channel = message.channel
                recent = await self._fetch_recent_context(channel)
                text = await self._generate_text(recent=recent, spontaneous=False)
                if not text:
                    return
                sent = await self._send_text_parts(channel=channel, parts_text=text, reply_to=message)
                self._record_bot_message(sent)
                self.state.last_reply_monotonic = time.monotonic()
                logger.info("Replied to mention")
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to reply to mention: %s", e)
            return

        # If the bot recently spoke, score whether this is engaging with it.
        score = self._score_engagement(message)
        if score > 0:
            self.state.author_scores[message.author.id] = self.state.author_scores.get(message.author.id, 0.0) + score
            total = self.state.author_scores[message.author.id]
            if total >= float(self.cfg.discord.engagement_threshold):
                if self._cooldown_ok():
                    try:
                        channel = message.channel
                        recent = await self._fetch_recent_context(channel)
                        text = await self._generate_text(recent=recent, spontaneous=False)
                        if not text:
                            return
                        sent = await self._send_text_parts(channel=channel, parts_text=text)
                        self._record_bot_message(sent)
                        self.state.last_reply_monotonic = time.monotonic()
                        # Extend engagement window on successful interaction.
                        self._enter_engagement()
                        logger.info("Engagement threshold met; responded (score=%.2f)", total)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("Failed threshold response: %s", e)
                return

        # Optional probabilistic replies only during engagement windows.
        if not self._in_engagement():
            return
        if not self._cooldown_ok():
            return
        if random.random() >= float(self.cfg.discord.reply_probability):
            return

        try:
            channel = message.channel
            recent = await self._fetch_recent_context(channel)
            text = await self._generate_text(recent=recent, spontaneous=False)
            if not text:
                return
            sent = await self._send_text_parts(channel=channel, parts_text=text, reply_to=message)
            self._record_bot_message(sent)
            self.state.last_reply_monotonic = time.monotonic()
            logger.info("Replied in engagement window")
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to reply: %s", e)


def resolve_discord_token(cfg: AppConfig) -> str:
    if cfg.discord.token:
        return cfg.discord.token
    env_key = cfg.discord.token_env or "DISCORD_TOKEN"
    token = os.environ.get(env_key, "")
    if not token:
        raise RuntimeError(f"Discord token missing: set discord.token or env var {env_key}")
    return token
