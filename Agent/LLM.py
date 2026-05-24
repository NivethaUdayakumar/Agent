from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"
MODEL_PATH_ENV_VAR = "LOCAL_GGUF_MODEL_PATH"
LEGACY_MODEL_NAME = "gemma3-1b-Q8_0.gguf"


def _list_gguf_models() -> list[Path]:
    if not MODEL_DIR.exists():
        return []

    return sorted(
        (
            path
            for path in MODEL_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".gguf"
        ),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )


def _default_model_path() -> Path:
    configured_model_path = os.getenv(MODEL_PATH_ENV_VAR, "").strip()

    if configured_model_path:
        return Path(configured_model_path).expanduser().resolve()

    legacy_model_path = MODEL_DIR / LEGACY_MODEL_NAME

    if legacy_model_path.exists():
        return legacy_model_path

    available_models = _list_gguf_models()

    if available_models:
        return available_models[0]

    return legacy_model_path


class LLMBackendError(RuntimeError):
    pass


def _format_load_error(model_path: Path, exc: Exception) -> str:
    error_text = str(exc).strip()
    normalized_error = error_text.lower()

    if "0xc000001d" in normalized_error or "illegal instruction" in normalized_error:
        return (
            f"Unable to load the local GGUF model at '{model_path}'. "
            "The installed llama.cpp backend is using CPU instructions that this machine does not support."
        )

    if error_text:
        return f"Unable to load the local GGUF model at '{model_path}'. Backend error: {error_text}"

    return f"Unable to load the local GGUF model at '{model_path}'."


def _format_missing_model_error(model_path: Path) -> str:
    available_models = _list_gguf_models()

    if available_models:
        listed_models = ", ".join(path.name for path in available_models)
        return (
            f"Model file not found at '{model_path}'. "
            f"Set {MODEL_PATH_ENV_VAR} to the exact file you want, or keep a GGUF model in '{MODEL_DIR}'. "
            f"Detected GGUF files: {listed_models}."
        )

    return (
        f"Model file not found at '{model_path}'. "
        f"Put a .gguf model inside '{MODEL_DIR}' or set {MODEL_PATH_ENV_VAR}."
    )


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    prompt_parts: list[str] = []

    for message in messages:
        role = str(message.get("role", "user")).strip().upper()
        content = str(message.get("content", "")).strip()

        if content:
            prompt_parts.append(f"{role}:\n{content}")

    prompt_parts.append("ASSISTANT:")
    return "\n\n".join(prompt_parts)


def _should_fallback_to_completion(exc: Exception) -> bool:
    error_text = str(exc).lower()

    return any(
        marker in error_text
        for marker in (
            "chat template",
            "chat format",
            "response_format",
            "does not support chat",
        )
    )


@dataclass
class LocalGGUFLLM:
    model_path: Path = field(default_factory=_default_model_path)
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    _backend: Any = field(default=None, init=False, repr=False)

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        if not self.model_path.exists():
            raise LLMBackendError(_format_missing_model_error(self.model_path))

        try:
            from langchain_community.llms import LlamaCpp

            self._backend = LlamaCpp(
                model_path=str(self.model_path),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                n_ctx=2048,
                n_batch=128,
                n_threads=2,
                verbose=False,
            )
        except Exception as exc:
            raise LLMBackendError(_format_load_error(self.model_path, exc)) from exc

    def _invoke_generation(self, messages: list[dict[str, str]]) -> str:
        prompt = _messages_to_prompt(messages)
        
        try:
            response = self._backend.invoke(prompt)
            return str(response).strip()
        except Exception as exc:
            raise LLMBackendError(f"Model generation failed: {exc}") from exc

    def invoke(self, messages: list[dict[str, str]]) -> str:
        self._ensure_backend()

        try:
            return self._invoke_generation(messages)
        except LLMBackendError:
            raise
        except Exception as exc:
            raise LLMBackendError(f"Local GGUF inference failed: {exc}") from exc


def create_llm() -> LocalGGUFLLM:
    return LocalGGUFLLM()
