from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    csv_path: Path
    output_jsonl: Path
    stats_json: Path
    style_profile_json: Path


@dataclass(frozen=True)
class TargetConfig:
    author_id: str


@dataclass(frozen=True)
class PreprocessConfig:
    min_chars: int = 5
    replace_urls: bool = True


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: str  # sentence_transformers | lm_studio
    model: str
    device: str = "cpu"
    batch_size: int = 128


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str = "http://127.0.0.1:1234"
    embeddings_model: str = ""


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float = 0.4
    max_tokens: int = 180
    top_p: float = 0.9
    stop: list[str] | None = None
    seed: int | None = 42
    retries: int = 3


@dataclass(frozen=True)
class DiscordConfig:
    token: str = ""
    token_env: str = "DISCORD_TOKEN"
    channel_id: int = 0

    spontaneous_interval_minutes: int = 30
    engagement_window_minutes: int = 8
    reply_probability: float = 0.15
    reply_when_mentioned: bool = True
    ignore_bots: bool = True
    context_messages: int = 30
    retrieval_k: int = 8
    cooldown_seconds: int = 8
    engagement_threshold: float = 2.0
    engagement_max_age_seconds: int = 900


@dataclass(frozen=True)
class IndexConfig:
    faiss_index_path: Path
    meta_jsonl_path: Path
    build_info_json: Path


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    target: TargetConfig
    preprocess: PreprocessConfig
    embeddings: EmbeddingsConfig
    lm_studio: LMStudioConfig
    generation: GenerationConfig
    discord: DiscordConfig
    index: IndexConfig


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing config key: {key}")
    return d[key]


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset = _require(raw, "dataset")
    target = _require(raw, "target")
    preprocess = raw.get("preprocess", {})
    embeddings = _require(raw, "embeddings")
    lm_studio = raw.get("lm_studio", {})
    generation = raw.get("generation", {})
    discord = raw.get("discord", {})
    index = _require(raw, "index")

    return AppConfig(
        dataset=DatasetConfig(
            csv_path=Path(_require(dataset, "csv_path")),
            output_jsonl=Path(_require(dataset, "output_jsonl")),
            stats_json=Path(_require(dataset, "stats_json")),
            style_profile_json=Path(dataset.get("style_profile_json", "./data/style_profile.json")),
        ),
        target=TargetConfig(author_id=str(_require(target, "author_id"))),
        preprocess=PreprocessConfig(
            min_chars=int(preprocess.get("min_chars", 5)),
            replace_urls=bool(preprocess.get("replace_urls", True)),
        ),
        embeddings=EmbeddingsConfig(
            provider=str(_require(embeddings, "provider")),
            model=str(_require(embeddings, "model")),
            device=str(embeddings.get("device", "cpu")),
            batch_size=int(embeddings.get("batch_size", 128)),
        ),
        lm_studio=LMStudioConfig(
            base_url=str(lm_studio.get("base_url", "http://127.0.0.1:1234")),
            embeddings_model=str(lm_studio.get("embeddings_model", "")),
        ),
        generation=GenerationConfig(
            model=str(generation.get("model", "")) or "qwen2.5-7b-instruct",
            temperature=float(generation.get("temperature", 0.4)),
            max_tokens=int(generation.get("max_tokens", 180)),
            top_p=float(generation.get("top_p", 0.9)),
            stop=list(generation.get("stop", [])) if generation.get("stop", None) is not None else None,
            seed=(None if generation.get("seed", None) in (None, "") else int(generation.get("seed"))),
            retries=int(generation.get("retries", 3)),
        ),
        discord=DiscordConfig(
            token=str(discord.get("token", "")),
            token_env=str(discord.get("token_env", "DISCORD_TOKEN")),
            channel_id=int(discord.get("channel_id", 0)),
            spontaneous_interval_minutes=int(discord.get("spontaneous_interval_minutes", 30)),
            engagement_window_minutes=int(discord.get("engagement_window_minutes", 8)),
            reply_probability=float(discord.get("reply_probability", 0.15)),
            reply_when_mentioned=bool(discord.get("reply_when_mentioned", True)),
            ignore_bots=bool(discord.get("ignore_bots", True)),
            context_messages=int(discord.get("context_messages", 30)),
            retrieval_k=int(discord.get("retrieval_k", 8)),
            cooldown_seconds=int(discord.get("cooldown_seconds", 8)),
            engagement_threshold=float(discord.get("engagement_threshold", 3.0)),
            engagement_max_age_seconds=int(discord.get("engagement_max_age_seconds", 900)),
        ),
        index=IndexConfig(
            faiss_index_path=Path(_require(index, "faiss_index_path")),
            meta_jsonl_path=Path(_require(index, "meta_jsonl_path")),
            build_info_json=Path(_require(index, "build_info_json")),
        ),
    )

