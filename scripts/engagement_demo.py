from __future__ import annotations

import argparse

from musinksbot.discord_bot import score_engagement_text


def main() -> None:
    p = argparse.ArgumentParser(description="Quick engagement scoring demo (no Discord connection).")
    p.add_argument("--bot-name", default="musinks")
    p.add_argument("--age", type=float, default=30.0, help="Seconds since the bot's last message")
    args = p.parse_args()

    cases = [
        ("lol", False, False),
        ("why do you do everything the hard way", False, False),
        ("musinks why do you do everything the hard way", False, False),
        ("@musinks why do you do everything the hard way", False, True),
        ("cheers pal", False, False),
        ("did you mean that?", False, False),
        ("cus its fun", False, False),
        ("because i can", False, False),
        ("nah youre wrong mate", True, False),
    ]

    for content, is_direct_reply, mentions_bot in cases:
        s_no_question = score_engagement_text(
            content=content,
            age_seconds=float(args.age),
            is_direct_reply=is_direct_reply,
            mentions_bot=mentions_bot,
            bot_name=str(args.bot_name),
            bot_asked_question=False,
            last_bot_text="ok probs gonna post about it soon",
        )
        s_question = score_engagement_text(
            content=content,
            age_seconds=float(args.age),
            is_direct_reply=is_direct_reply,
            mentions_bot=mentions_bot,
            bot_name=str(args.bot_name),
            bot_asked_question=True,
            last_bot_text="ok probs gonna post about it soon",
        )
        print(
            f"{s_no_question:>4.1f} / {s_question:>4.1f}  reply={is_direct_reply} mention={mentions_bot}  {content}"
        )


if __name__ == "__main__":
    main()
