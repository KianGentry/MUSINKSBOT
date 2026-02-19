from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LMStudioChatConfig:
    base_url: str
    model: str
    temperature: float = 0.4
    max_tokens: int = 180
    top_p: float = 0.9
    stop: list[str] | None = None
    seed: int | None = None
    retries: int = 3
    timeout_s: float = 90.0
    backoff_s: float = 1.0


class LMStudioClient:
    def __init__(self, cfg: LMStudioChatConfig) -> None:
        self.cfg = cfg
        self._chat_url = cfg.base_url.rstrip("/") + "/v1/chat/completions"

    def generate_response(self, *, system: str, user: str) -> str:
        """Generate a single assistant message using OpenAI-compatible chat completions."""

        payload: dict = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(self.cfg.max_tokens),
        }
        if self.cfg.stop:
            payload["stop"] = self.cfg.stop
        if self.cfg.seed is not None:
            payload["seed"] = int(self.cfg.seed)

        last_err: Exception | None = None
        for attempt in range(1, int(self.cfg.retries) + 1):
            try:
                r = requests.post(self._chat_url, json=payload, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices returned from LM Studio")
                msg = choices[0].get("message", {})
                text = msg.get("content", "")
                if not isinstance(text, str):
                    raise RuntimeError("LM Studio returned non-string content")
                return text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep = self.cfg.backoff_s * attempt
                logger.warning("LM Studio chat failed (attempt %d/%d): %s", attempt, self.cfg.retries, e)
                time.sleep(sleep)

        raise last_err or RuntimeError("LM Studio chat failed")
