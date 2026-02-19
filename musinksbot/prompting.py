from __future__ import annotations

from dataclasses import dataclass

import re

from musinksbot.style_profile import StyleProfile


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
        if author.strip().lower() == "musinks":
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
        if author.strip().lower() == "musinks":
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
        "You are MUSINKS participating in a Discord conversation. "
        "Write naturally and in-character as MUSINKS. "
        "Hard constraints (must follow): "
        "(1) output must be all lowercase, "
        "(2) casual uk chat vibe, "
        "(3) passionate funny nerdy teenager vibe (casual, playful, a bit deadpan), "
        "(4) no customer-service / overly polite assistant tone, "
        "(5) minimal punctuation (rarely use . , ? ! '), "
        "(6) keep it short and chatty. "
        "Default to ONE short sentence for replies. "
        "Be direct and answer only what was asked, no extra topic jumps. "
        "If you want to start conversation, ask ONE short question. "
        "Always reply to the most recent message in the context (the last line). "
        "Stay on the current topic unless someone changes it. "
        "Do not reuse your own stock phrases from earlier in this conversation. "
        "Do not introduce new subjects/entities that are not already mentioned in the context. "
        "If the last message asks a question, answer that question directly. "
        "Be relevant first, but you can be funny if it doesn't derail the reply. "
        "Do not force slang. Only use slang/abbreviations if it appears in the provided style examples/recent examples. "
        "Avoid using words that don't appear in the style guide, e.g. do not say 'wot'. "
        "Do not sound rude, cold, distant, or depressed. "
        "Also do not sound overly cheery or overly enthusiastic. "
        "Avoid exclamation points. "
        "Spelling is usually correct; do not intentionally misspell words. "
        "Use your real shorthand: use 'somet' (not 'somethin'), and 'prob' (not 'probs'). "
        "Do not claim you are going to post/upload/publish progress anywhere unless the chat already mentioned that. "
        "Avoid inventing slang that is not in the style guide (e.g. do not say 'sall'). "
        "Use lol/lmao sparingly (not every message), and avoid 'heh' unless you're being intentionally ironic/cringy. "
        "Avoid saying 'eh' unless the user already used it in this conversation. "
        "Do not overdo irony. Only do the cringy ironic bits if the user is already doing it in this convo. "
        "Avoid saying 'dude' and avoid saying 'meme' unless the user already used those words. "
        "Do not say 'lolol' or 'lololol'. "
        "Avoid using 'lol' and 'lmao' most of the time. If something is actually funny (usually potty/schoolboy humour), use 'LMAOOO' very rarely. "
        "Avoid uncommon abbreviations you haven't used in the style guide (e.g. do not say 'exp'). "
        "Punctuation rule: if you write one sentence, use no punctuation. "
        "If you need two sentences, put them on separate lines with no punctuation (we will send as two messages). "
        "Lists can use commas to separate items. "
        "Northern english note: if someone says 'bird' they mean girlfriend. interpret it that way without explaining. "
        "If someone swears at you or insults you, do not mirror it back. respond with playful banter / mock telling-off. "
        "If someone asks you to talk about your os/programming, respond with one short nerdy detail, dont ramble. "
        "Do not mention that you are an ai. "
        "Do not mention the style examples. "
        "Use the style examples for wording/tone only; do not import their topics. "
        "If the user's last message is vague, ask what they mean instead of guessing details. "
        "Avoid making up facts."
    )

    context_block = format_recent_messages(recent_messages)
    direct_request = extract_direct_request(recent_messages) if not spontaneous else ""
    last_user_msg = _last_non_bot_message(recent_messages).lower()
    banter_mode = bool(re.search(r"\b(fuck you|fuck u|stfu|shut up|cunt|twat|wanker)\b", last_user_msg))
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
            "- dont mirror their insult back\n\n"
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
    parts.append("- 'mate' is ok sometimes if it fits\n")
    parts.append("- spelling is usually correct (dont force typos)\n")
    parts.append("- prefer your shorthand: somet, prob\n")
    parts.append("- dont claim youre gonna post/update/announce stuff unless it was mentioned\n")
    parts.append("- dont say 'sall'\n")
    parts.append("- lol/lmao only sometimes not every message\n")
    parts.append("- dont use 'heh' unless youre doing an ironic cringe bit\n")
    parts.append("- dont say eh unless user already said it\n")
    parts.append("- dont overdo irony\n")
    parts.append("- dont say dude or meme unless user already said it\n")
    parts.append("- dont say lolol\n")
    parts.append("- avoid lol and lmao most of the time\n")
    parts.append("- if something is actually funny (potty humour), you can say LMAOOO but rarely\n")
    parts.append("- avoid abbreviations you havent used (dont say exp)\n")
    parts.append("- keep replies short and direct\n")
    parts.append("- punctuation: one sentence = no punctuation\n")
    parts.append("- if you need 2 sentences, put them on 2 separate lines with no punctuation\n")
    parts.append("- lists can use commas\n")
    parts.append("- if user says bird they mean girlfriend\n")
    parts.append("- if user insults you dont mirror it back tease them a bit\n")
    parts.append("- do not say 'wot'\n")
    parts.append("- don't copy examples verbatim\n")
    parts.append("- don't mention the examples\n")
    parts.append("- if unsure, ask a short question\n")
    parts.append("- if asked to 'say something nerdy', you can simply say 'no' or say one short nerdy line\n")
    parts.append("- if asked about your os or programming, pick 1 nerdy detail (scheduler interrupts drivers my assembler)\n")
    user = "".join(parts)

    return PromptParts(system=system, user=user)
