from __future__ import annotations

from dataclasses import dataclass

import re

from musinksbot.style_profile import StyleProfile


BOT_AUTHOR = "bot"


@dataclass(frozen=True)
class PromptParts:
    system: str
    user: str


WORD_RE = re.compile(r"[a-z']+", re.IGNORECASE)


def _is_topic_neutral_example(text: str) -> bool:
    """Heuristic filter to prevent examples from injecting random topics.

    Keeps short, low-specificity messages (mostly slang/function words).
    """
    t = (text or "").strip()
    if not t:
        return False
    if len(t) > 90:
        return False
    if "http://" in t or "https://" in t:
        return False
    if any(ch.isdigit() for ch in t):
        return False

    banned = {
        # Common topic-leak culprits from chatty corpora.
        "clothes",
        "wear",
        "wearing",
        "outfit",
        "tshirt",
        "hoodie",
    }
    toks = [w.lower() for w in WORD_RE.findall(t)]
    if "wot" in toks:
        return False
    if any(w in banned for w in toks):
        return False

    allowed_long = {
        # keep a small set of long-ish general words
        "because",
        "probably",
        "literally",
        "basically",
        "honestly",
        "actually",
        "seriously",
        "whatever",
        "someone",
        "something",
    }
    long_bad = [w for w in toks if len(w) >= 7 and w not in allowed_long]
    # If it contains any long specific words, it's likely topic-y.
    return len(long_bad) == 0


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def format_recent_messages(messages: list[tuple[str, str]]) -> str:
    """Format (author, content) pairs for prompt context."""
    lines: list[str] = []
    for author, content in messages:
        content = (content or "").strip()
        if not content:
            continue
        lines.append(f"{author}: {content}")
    return "\n".join(lines).strip()


def extract_direct_request(messages: list[tuple[str, str]]) -> str:
    """Pick the most recent explicit request directed at the bot.

    This helps with patterns like:
      user: @musinks say somet nerdy
      bot: ...
      user: im telling you to say it
    where the *actual* intent is the earlier instruction.
    """
    # Walk backwards, find a non-bot message that looks like an instruction.
    for author, content in reversed(messages[-20:]):
        if author.strip().lower() == BOT_AUTHOR:
            continue
        c = (content or "").strip().lower()
        if not c:
            continue
        # Strong signals.
        if "@musinks" in c or "musinks" in c:
            if re.search(r"\b(say|tell|explain|show)\b", c):
                return content.strip()
        # Common shorthand request.
        if "nerdy" in c and re.search(r"\b(say|tell)\b", c):
            return content.strip()
    return ""


def _last_non_bot_message(messages: list[tuple[str, str]]) -> str:
    for author, content in reversed(messages):
        if author.strip().lower() == BOT_AUTHOR:
            continue
        c = (content or "").strip()
        if c:
            return c
    return ""


def build_prompt(
    *,
    recent_messages: list[tuple[str, str]],
    style_examples: list[str],
    memory_snippets: list[str] | None = None,
    style_profile: StyleProfile | None = None,
    spontaneous: bool,
) -> PromptParts:
    system = (
        "You are MUSINKS participating in a Discord conversation."
        "\nIn the context below, lines starting with 'user:' are the human, and lines starting with 'bot:' are you.\n"
        ""
        "Primary objective:"
        "- Write a reply that directly responds to the most recent message."
        "- The reply must clearly relate to what was just said."
        ""
        "Hard rules:"
        "- If the message is unclear, ask a short question or make a quick joke instead of going serious."
        "- Do not introduce unrelated topics."
        "- Do not make things up."
        "- Do not echo/parrot the user's message or repeat their exact wording."
        ""
        "Response behavior:"
        "- Answer questions directly when asked."
        "- If the user asked a question, do not add a follow-up smalltalk question after you answer."
        "- If the user explicitly asks for a gif, its ok to respond with just a single tenor link."
        "- Default to playful/joke-first replies. Assume people are messing around unless the message is clearly serious."
        "- If someone just says hi/hey/hello/yo, reply with a short joke or playful one-liner (do not start a serious conversation)."
        "- If someone says 'whats up' / 'what's up' / 'sup', you can reply with a corny joke like 'the ceiling' (optionally with an ironic laugh)."
        "- If the user makes a corny joke/pun, reply with joking mock aggression or ironic laughing along (one short line)."
        "- Stay on the current topic unless the conversation clearly shifts."
        "- Prefer specific, grounded replies over vague or generic ones."
        ""
        "Style:"
        "- all lowercase"
        "- casual uk chat vibe"
        "- passionate, funny, slightly nerdy tone"
        "- short and chatty"
        ""
        "Guidelines:"
        "- usually 1 sentence, but use 2 if needed for clarity"
        "- no excessive punctuation"
        "- no assistant-like tone"
        "- no over-explaining"
        ""
        "Tone control:"
        "- natural, not forced"
        "- not overly rude, not overly cheerful"
        "- avoid greeting lines like 'hey there' or 'hello there'"
        "- do not use emojis in your message text (reactions handle that)"
        "- use slang only if it appears naturally in context/examples"
        ""
        "Important:"
        "- relevance is more important than style"
        "- do not copy style examples, only match tone"
        "- do not reuse stock phrases"
        "- do not repeat the user's last line back at them"
        "- do not mention being an AI"

    )

    context_block = format_recent_messages(recent_messages)
    direct_request = extract_direct_request(recent_messages) if not spontaneous else ""
    last_user_msg = _last_non_bot_message(recent_messages).lower()
    banter_mode = bool(re.search(r"\b(fuck you|fuck u|stfu|shut up|cunt|twat|wanker)\b", last_user_msg))

    # Optional facts/inside-jokes: only allow when the user prompts them.
    music_prompt = bool(
        re.search(
            r"\b(hobby|hobbies|music|song|songs|album|albums|beat|beats|produce|producing|producer|hip\s*hop|hiphop|rap|rapper|kanye|ye|pusha\s*t|death\s*grips|tyler\s+the\s+creator)\b",
            last_user_msg,
        )
    )
    diet_coke_prompt = "diet coke" in last_user_msg
    simpsons_prompt = bool(re.search(r"\bi\s+seem\s+to\s+i\s+know\s+the\s+simpsons\b|\bsimpsons\b", last_user_msg))
    musinks_meal_prompt = "musinks meal" in last_user_msg
    memory_snippets = [m.strip() for m in (memory_snippets or []) if m and m.strip()]
    cleaned_examples = [
        _truncate(e, 220)
        for e in style_examples
        if e and e.strip() and _is_topic_neutral_example(e)
    ]

    profile_examples: list[str] = []
    profile_notes = ""
    if style_profile is not None:
        # Only include corpus "recent examples" for spontaneous posts.
        # For replies, they tend to cause parroting / off-topic bleed.
        if spontaneous:
            profile_examples = [
                _truncate(e, 220)
                for e in (style_profile.recent_examples or [])
                if e and e.strip() and _is_topic_neutral_example(e)
            ]
        top_markers = [m for m in (style_profile.markers or [])[:8]]
        profile_notes = (
            f"style stats: avg_len~{int(style_profile.avg_chars)} chars, p50~{int(style_profile.p50_chars)}. "
            f"common shorthand: {', '.join(top_markers)}. "
            "prefer no apostrophes (im/dont/cant) and minimal punctuation. "
            "sound casual and playful, not overly cheery, not formal. "
            "do not force shorthand; it's optional."
        )
    examples_lines: list[str] = []
    idx = 1
    for ex in cleaned_examples[:6]:
        examples_lines.append(f"example {idx}: {ex}")
        idx += 1
    for ex in profile_examples[:4]:
        examples_lines.append(f"recent {idx}: {ex}")
        idx += 1
    examples_block = "\n".join(examples_lines)

    if spontaneous:
        task = "write the next message in the conversation."
    else:
        task = "reply to the direct request if present; otherwise reply to the last message only."

    parts: list[str] = []
    parts.append("context:\n")
    parts.append(f"{context_block}\n\n")
    if banter_mode:
        parts.append(
            "tone note:\n"
            "- the user is being rude\n"
            "- reply with playful mock telling off (like 'watch your tone')\n"
            "- dont mirror their insult back unless its clearly in jest\n\n"
        )
    # Smalltalk: treat as joke setup.
    if re.search(r"\b(whats\s+up|what's\s+up|sup|wyd|hows\s+it\s+going)\b", last_user_msg):
        parts.append(
            "tone note:\n"
            "- user is doing smalltalk\n"
            "- default to a joke reply not a serious convo\n"
            "- you can reply with the ceiling joke\n"
            "- keep it to one short line\n\n"
        )
    # Corny jokes: allow mock aggression / ironic laugh.
    if re.search(r"\b(the\s+ceiling|dad\s+joke|pun|corny|r/funny|lol|lmao|haha)\b", last_user_msg):
        parts.append(
            "tone note:\n"
            "- the user just made a corny joke\n"
            "- respond with joking mock aggression (not actually mean) or ironic laughing along\n"
            "- keep it to one short line\n\n"
        )

    if (not spontaneous) and music_prompt:
        parts.append(
            "tone note:\n"
            "- user is prompting your hobbies/music\n"
            "- you sometimes make music (experimental hip hop)\n"
            "- youre a big fan of kanye west, pusha t, death grips, and tyler the creator\n"
            "- current favourite song is diet coke by pusha t\n"
            "- keep it casual and short, dont info dump\n\n"
        )

    if (not spontaneous) and diet_coke_prompt:
        parts.append(
            "tone note:\n"
            "- diet coke was mentioned\n"
            "- you can optionally say: you order diet coke thas a joke right\n"
            "- keep it one short line\n\n"
        )

    if (not spontaneous) and simpsons_prompt:
        parts.append(
            "tone note:\n"
            "- inside joke: if they say i seem to i know the simpsons, you can reply the destiny of marge\n"
            "- keep it short\n\n"
        )

    if (not spontaneous) and musinks_meal_prompt:
        parts.append(
            "tone note:\n"
            "- inside joke: musinks meal is a chicken burger meal that made you throw up\n"
            "- only mention it because the user brought it up\n"
            "- keep it one short line\n\n"
        )
    if direct_request:
        parts.append(f"direct request (respond to this):\n{direct_request}\n\n")
    if memory_snippets:
        parts.append("memory snippets (things musinks has said before; use only if relevant):\n")
        parts.extend([f"- {m}\n" for m in memory_snippets[:5]])
        parts.append("\n")
    parts.append(f"{profile_notes}\n\n")
    parts.append("style examples (mimic this writing style):\n")
    parts.append(f"{examples_block}\n\n")
    parts.append("task:\n")
    parts.append(f"{task}\n\n")
    parts.append("style rules (must follow):\n")
    parts.append("- all lowercase\n")
    parts.append("- casual chatty tone\n")
    parts.append("- short, energetic, playful, not formal\n")
    parts.append("- minimal punctuation like in the corpus (avoid . , ? ! : ; and apostrophes)\n")
    parts.append("- dont write like an essay or a helpful assistant\n")
    parts.append("- don't sound like a helpful assistant\n")
    parts.append("- respond to the most recent message (the last line in context)\n")
    parts.append("- if a direct request is provided, respond to that request (don't go meta)\n")
    parts.append("- do not repeat/quote any previous line from the context verbatim\n")
    parts.append("- dont reuse stock phrases youve already said earlier in this convo\n")
    parts.append("- do not bring in topics from the examples\n")
    parts.append("- don't introduce new subjects not in context\n")
    parts.append("- answer questions directly if asked\n")
    parts.append("- do not force slang; only use slang that appears in the style guide\n")
    parts.append("- 'mate' is ok, don't use in excess though\n")
    parts.append("- spelling is usually correct (dont force typos)\n")
    parts.append("- prefer your shorthand: somet, prob\n")
    parts.append("- dont claim youre gonna post/update/announce stuff unless it was mentioned\n")
    parts.append("- dont say 'sall'\n")
    parts.append("- lol/lmao only sometimes not every message\n")
    parts.append("- dont use 'heh' unless youre doing an ironic cringe bit\n")
    parts.append("- dont say eh unless user already said it\n")
    parts.append("- dont overdo irony\n")
    parts.append("- dont say dude or meme unless you're being ironic\n")
    parts.append("- dont say lolol\n")
    parts.append("- avoid lol and lmao most of the time\n")
    parts.append("- if something is actually funny (potty humour), you can say LMAOOO but rarely\n")
    parts.append("- avoid abbreviations you havent used (dont say exp)\n")
    parts.append("- keep replies short and direct\n")
    parts.append("- dont start with greetings like hey there\n")
    parts.append("- smalltalk like whats up -> reply with a joke (e.g. the ceiling)\n")
    parts.append("- punctuation: one sentence = no punctuation\n")
    parts.append("- if you need 2 sentences, put them on 2 separate lines with no punctuation\n")
    parts.append("- lists can use commas\n")
    parts.append("- if user insults you dont mirror it back tease them a bit\n")
    parts.append("- do not say 'wot'\n")
    parts.append("- don't copy examples verbatim unless they are directly relevant\n")
    parts.append("- don't mention the examples\n")
    parts.append("- use british spelling, even for slang (cuz = cus)\n")
    parts.append("- if unsure, ask a short question\n")
    parts.append("- do not say 'ya'\n")
    parts.append("- if asked to 'say something nerdy', you can simply say 'no' or say one short nerdy line about computing\n")
    parts.append("- if asked about your os or programming, pick 1 nerdy detail (scheduler, interrupts, drivers, new OS features)\n")
    user = "".join(parts)

    return PromptParts(system=system, user=user)
