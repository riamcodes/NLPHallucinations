"""llama.cpp-backed QA models (Mistral, Llama-3, etc.)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - import guard
    Llama = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .base import QAModel

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp style models."""

    model_path: Path
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = (
        "You are a factual assistant. Only answer using information supported by the "
        "provided context. If unsure, state that the answer cannot be derived."
    )


class LlamaCppQA(QAModel):
    """Generic llama.cpp wrapper suitable for Mistral and Llama-3 GGUF weights."""

    def __init__(self, name: str, config: LlamaCppConfig):
        super().__init__(name=name)
        self.config = config
        self._llm: Optional[Llama] = None

    def load(self) -> None:
        """Initialise the llama.cpp model."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required to load GGUF models."
            ) from _IMPORT_ERROR

        logger.info("Loading llama.cpp model %s from %s", self.name, model_path)
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
        )

    def answer(self, question: str, context: str) -> Dict[str, Any]:
        if not self._llm:
            return self.mark_failed("Model not loaded yet.")

        messages = list(self.config.chat_history)
        messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Use the supplied context to answer the question. "
                    "If the context lacks the answer, reply with 'Unknown'.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}"
                ),
            }
        )

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )

        choice = response["choices"][0]["message"]["content"].strip()

        return {
            "model": self.name,
            "answer": choice,
            "metadata": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }

