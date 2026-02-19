# MUSINKSBOT (local)

Local Discord bot utilities to emulate a target user's writing style using retrieval-based style injection (no fine-tuning, no reply-pair reconstruction).

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit `config.yaml` and set `target.author_id`.

### LM Studio embeddings (recommended)

If you're using LM Studio, ensure it exposes an OpenAI-compatible `POST /v1/embeddings` endpoint and set:

- `embeddings.provider: lm_studio`
- `embeddings.model: nomic-embed-text-v1.5` (or whichever embedding model you loaded in LM Studio)
- `lm_studio.base_url: http://127.0.0.1:1234`

Optional quick connectivity test:

```bash
python scripts/lmstudio_smoke.py --base-url http://127.0.0.1:1234 --model nomic-embed-text-v1.5
```

## 2) Preprocess the CSV

```bash
python scripts/preprocess.py --config config.yaml
```

Outputs:
- `data/clean_messages.jsonl`
- `data/preprocess_stats.json`

## 3) Build embeddings + FAISS index

```bash
python scripts/build_index.py --config config.yaml
```

Outputs:
- `data/style_index.faiss`
- `data/style_index_meta.jsonl`
- `data/style_index_build_info.json`

## 4) Quick retrieval smoke test

```bash
python scripts/retrieve_demo.py --config config.yaml --text "ive been programming an assembler"
```

## 5) Run the Discord bot

1) In `config.yaml`, set:

- `discord.channel_id` (target channel)
- Either `discord.token` OR set env var `DISCORD_TOKEN`
- `generation.model` to your LM Studio chat model id (e.g. `qwen2.5-7b-instruct`)

2) Ensure your Discord application has **Message Content Intent** enabled (Developer Portal → Bot → Privileged Gateway Intents).

3) Start the bot:

```bash
python scripts/run_bot.py --config config.yaml
```

Behavior:
- Every ~30 minutes, posts a spontaneous message to `discord.channel_id`.
- After posting, tracks follow-up messages and uses an engagement score; once `discord.engagement_threshold` is met (within `discord.engagement_max_age_seconds`), it replies as a normal channel message.

Optional scoring demo (no Discord connection): run `python scripts/engagement_demo.py --age 45`.
