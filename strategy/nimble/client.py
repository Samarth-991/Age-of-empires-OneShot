"""Minimal client for Ollama's decision-model endpoint (``POST /v1/systemone``)."""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import (
    NIMBLE_BATCH_SIZE,
    NIMBLE_KEEP_ALIVE,
    NIMBLE_TIMEOUT_SECS,
    OLLAMA_BASE_URL,
    OLLAMA_NIMBLE_MODEL,
)

MAX_QUESTIONS_PER_REQUEST = 64


class NimbleError(RuntimeError):
    """Nimble could not be reached or returned an unusable response."""


@dataclass(frozen=True)
class DecisionResponse:
    answers: dict[str, dict[str, Any]]
    latency_ms: float
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class DecisionClient(Protocol):
    def decide(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> DecisionResponse: ...


class NimbleClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_NIMBLE_MODEL,
        timeout_secs: float = NIMBLE_TIMEOUT_SECS,
        keep_alive: str | None = NIMBLE_KEEP_ALIVE,
        batch_size: int = NIMBLE_BATCH_SIZE,
    ) -> None:
        if not 1 <= batch_size <= MAX_QUESTIONS_PER_REQUEST:
            raise ValueError(f"batch_size must be between 1 and {MAX_QUESTIONS_PER_REQUEST}")
        self.url = f"{base_url.rstrip('/')}/v1/systemone"
        self.model = model
        self._timeout = timeout_secs
        self._keep_alive = keep_alive
        self._batch_size = batch_size

    def decide(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> DecisionResponse:
        """Answer all questions, splitting them into batches of ``batch_size``."""
        if not questions:
            raise ValueError("questions must not be empty")

        keys = list(questions)
        answers: dict[str, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        model = self.model
        t0 = time.perf_counter()
        for start in range(0, len(keys), self._batch_size):
            batch = {k: questions[k] for k in keys[start:start + self._batch_size]}
            raw = self._post(state, batch)
            answers.update(raw.get("answers") or {})
            model = str(raw.get("model") or model)
            for name, count in (raw.get("usage") or {}).items():
                usage[name] = usage.get(name, 0) + int(count)
        return DecisionResponse(
            answers=answers,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=model,
            usage=usage,
        )

    def _post(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise NimbleError(f"Nimble HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise NimbleError(f"Cannot reach Ollama at {self.url}: {exc.reason}") from exc
        except (http.client.HTTPException, OSError, json.JSONDecodeError) as exc:
            raise NimbleError(f"Nimble request failed: {type(exc).__name__}: {exc}") from exc
