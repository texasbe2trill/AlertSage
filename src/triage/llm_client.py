from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Optional debug flag shared with the rest of the project
LLM_DEBUG = os.getenv("NLP_TRIAGE_LLM_DEBUG", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}


try:  # pragma: no cover - import is environment dependent
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - if llama_cpp is not installed
    Llama = None  # type: ignore


def _debug(msg: str) -> None:
    """Lightweight debug logger for LLM operations."""
    if LLM_DEBUG:
        print(f"[LLM CLIENT] {msg}", flush=True)


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

        # llama_cpp returns a dict with `choices[0]["text"]`
        text = result.get("choices", [{}])[0].get("text", "")
        return str(text).strip()

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


__all__ = ["LocalLLMClient"]
