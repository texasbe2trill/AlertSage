from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import requests

# Optional debug flag shared with the rest of the project
LLM_DEBUG = os.getenv("NLP_TRIAGE_LLM_DEBUG", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}

HF_DEFAULT_MODEL = (
    os.getenv("TRIAGE_HF_MODEL")
    or os.getenv("HF_MODEL")
    or "meta-llama/Llama-3.1-8B-Instruct:cerebras"
)
HF_TOKEN_ENV = os.getenv("TRIAGE_HF_TOKEN") or os.getenv("HF_TOKEN") or ""


try:  # pragma: no cover - import is environment dependent
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - if llama_cpp is not installed
    Llama = None  # type: ignore


def _debug(msg: str) -> None:
    """Lightweight debug logger for LLM operations."""
    if LLM_DEBUG:
        print(f"[LLM CLIENT] {msg}", flush=True)


def _get_streamlit_secrets() -> tuple[str, str]:
    """Safely read HF credentials from Streamlit secrets when available."""
    try:
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if secrets:
            token = str(secrets.get("HF_TOKEN", "")).strip()
            model = str(secrets.get("HF_MODEL", "")).strip()
            return model, token
    except Exception:
        pass
    return "", ""


def resolve_hf_credentials(
    model: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[str, str, bool]:
    """Resolve Hugging Face model/token from secrets → env → arguments.

    Returns (model, token, token_available).
    """

    secrets_model, secrets_token = _get_streamlit_secrets()

    resolved_token = (token or secrets_token or HF_TOKEN_ENV).strip()
    resolved_model = (model or secrets_model or HF_DEFAULT_MODEL).strip()

    if not resolved_model:
        resolved_model = HF_DEFAULT_MODEL

    # Hugging Face Router expects a repo id (optionally with provider suffix), not a local GGUF path
    if resolved_model.lower().endswith(".gguf"):
        _debug(
            "HF model appears to be a GGUF filename; use a repo id like 'meta-llama/Llama-3.1-8B-Instruct:cerebras'."
        )

    return resolved_model, resolved_token, bool(resolved_token)


@dataclass
class RateLimiter:
    """Sliding-window rate limiter used for hosted LLM calls."""

    max_requests: int = 5
    window_seconds: int = 60

    def __post_init__(self) -> None:
        self._events: deque[datetime] = deque()

    def check(self) -> Tuple[bool, float]:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        while self._events and self._events[0] < window_start:
            self._events.popleft()

        if len(self._events) >= self.max_requests:
            retry_after = self.window_seconds - (now - self._events[0]).total_seconds()
            return False, max(retry_after, 0.0)

        self._events.append(now)
        return True, 0.0


@dataclass
class HuggingFaceInferenceClient:
    """Minimal HF Router client with chat completions and rate limiting."""

    model: str
    token: str
    endpoint: str = "https://router.huggingface.co"
    timeout: int = 120
    max_prompt_chars: int = 8000
    max_new_tokens: int = 512
    temperature: float = 0.05
    rate_limiter: Optional[RateLimiter] = None

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("HuggingFaceInferenceClient requires a token")
        if not self.model:
            raise ValueError("HuggingFaceInferenceClient requires a model id")

        self.endpoint = self.endpoint.rstrip("/")
        self._session = requests.Session()
        _debug(
            f"Initialised HF Router client for model='{self.model}' at endpoint='{self.endpoint}'"
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except Exception:
                _debug("HF parse: failed to decode extracted JSON snippet")
        _debug("HF parse: no JSON found, returning empty dict")
        return {}

    def generate_json(self, prompt: str, *, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        prompt_to_send = prompt if len(prompt) <= self.max_prompt_chars else prompt[: self.max_prompt_chars]
        if len(prompt) > self.max_prompt_chars:
            _debug(
                f"Prompt truncated from {len(prompt)} to {len(prompt_to_send)} characters for HF inference."
            )

        if self.rate_limiter:
            allowed, retry_after = self.rate_limiter.check()
            if not allowed:
                raise RuntimeError(
                    f"Rate limit exceeded: wait {retry_after:.0f}s before retrying Hugging Face inference."
                )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_to_send}],
            "max_tokens": max_tokens or self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        url = f"{self.endpoint}/v1/chat/completions"
        _debug(
            f"Calling HF Router: url='{url}', max_tokens={payload['max_tokens']}, temperature={payload['temperature']}"
        )
        response = self._session.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code == 401:
            raise PermissionError("Hugging Face token rejected (401 Unauthorized)")
        if response.status_code == 429:
            detail = (
                response.json().get("error")
                if response.headers.get("content-type", "").startswith("application/json")
                else response.text
            )
            raise RuntimeError(f"Hugging Face rate limit hit (429): {detail}")
        if response.status_code >= 500:
            raise RuntimeError(f"Hugging Face service error {response.status_code}: {response.text}")
        if response.status_code >= 400:
            raise RuntimeError(
                f"Hugging Face request failed {response.status_code}: {response.text}"
            )

        data = response.json()
        _debug(f"HF raw response: {str(data)[:500]}")

        generated_text = ""
        try:
            generated_text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "") or ""
            )
        except Exception:
            _debug("HF response parsing failed; returning raw JSON")
            return data if isinstance(data, dict) else {}

        if not generated_text:
            _debug("HF response did not include message content; returning raw JSON")
            return data if isinstance(data, dict) else {}

        return self._parse_json_from_text(generated_text)


@dataclass
class LocalLLMClient:
    """Thin wrapper around a local llama.cpp-compatible model.

    This client is intentionally minimal so it can be used from both:
    - the CLI (for second-opinion classification), and
    - the synthetic generator (for lightly rewriting narratives).

    It exposes two main methods:
    - generate_text(): generic completion used by the generator.
    - generate_json(): completion with best-effort JSON extraction, used
      by the CLI for structured second-opinion outputs.
    """

    # Accept either `backend` or `model_path` so callers have flexibility
    backend: Optional[str] = None
    model_path: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024  # Increased for GPU acceleration - richer responses
    system_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalise model path
        path = self.backend or self.model_path
        if not path:
            raise ValueError(
                "LocalLLMClient requires a model path via `backend` or `model_path`."
            )

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LLM model not found at: {path}")

        if Llama is None:
            raise RuntimeError(
                "llama_cpp is not installed or importable. "
                "Install `llama-cpp-python` to enable LocalLLMClient."
            )

        self.model_path = path
        _debug(f"Initialising Llama backend with model_path='{self.model_path}'")

        # Keep the config conservative so we don't surprise the user.
        # If you want custom params, you can always add them later.
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            logits_all=False,
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, user_prompt: str) -> str:
        """Attach a system prompt if provided.

        The rest of the codebase currently uses the completion API
        without a chat template, so we keep this as a single string.
        """
        if self.system_prompt:
            # Lightweight separation so the model can distinguish roles.
            return (
                "System: "
                + self.system_prompt.strip()
                + "\n\n"
                + "User: "
                + user_prompt.strip()
                + "\nAssistant: "
            )
        return user_prompt

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Run a raw text completion and return the model's text.

        This is the primary entrypoint used by the synthetic generator
        to lightly rewrite descriptions.
        """
        mt = max_tokens if max_tokens is not None else self.max_tokens
        temp = self.temperature if temperature is None else temperature
        stop = stop or []

        full_prompt = self._build_prompt(prompt)
        _debug(
            f"Calling LLM.generate_text len(prompt)={len(full_prompt)}, "
            f"max_tokens={mt}, temperature={temp}"
        )

        result = self._llm(
            full_prompt,
            max_tokens=mt,
            temperature=temp,
            stop=stop,
        )

        # llama_cpp returns a dict with `choices[0]["text"]`; some builds may stream
        if isinstance(result, dict):
            text = result.get("choices", [{}])[0].get("text", "")
            return str(text).strip()

        # Streaming iterator fallback
        try:
            first_chunk = next(iter(result))
            if isinstance(first_chunk, dict):
                text = first_chunk.get("choices", [{}])[0].get("text", "")
                return str(text).strip()
            return str(first_chunk).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # JSON-focused helper used by CLI second-opinion logic
    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate text then try to extract a JSON object from it.

        The calling code is responsible for validating keys / schema.
        On failure, this returns an empty dict instead of raising.
        """
        raw = self.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # First attempt: direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Second attempt: extract first {...} block
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            _debug(f"generate_json: no JSON object found in output: {raw!r}")
            return {}

        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception as exc:  # pragma: no cover - highly model dependent
            _debug(
                "generate_json: failed to parse extracted JSON snippet "
                f"due to {exc!r}; snippet={snippet!r}"
            )
            return {}


__all__ = [
    "LocalLLMClient",
    "HuggingFaceInferenceClient",
    "RateLimiter",
    "resolve_hf_credentials",
]
