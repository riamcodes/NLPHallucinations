"""FLAN-T5 based QA backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

from .base import QAModel

logger = logging.getLogger(__name__)


@dataclass
class FlanConfig:
    """Configuration for loading a FLAN-T5 model."""

    model_name: str = "google/flan-t5-base"
    device: Optional[str] = None  # "cpu", "cuda", "mps"
    max_input_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9


class FlanT5QA(QAModel):
    """Sequence-to-sequence QA implementation using FLAN-T5."""

    def __init__(self, name: str = "flan-t5-base", config: Optional[FlanConfig] = None):
        super().__init__(name=name)
        self.config = config or FlanConfig()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None

    def load(self) -> None:
        """Load tokenizer and model weights."""
        logger.info("Loading FLAN-T5 model %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

        if self.config.device:
            logger.info("Moving FLAN-T5 to device %s", self.config.device)
            self._model.to(self.config.device)

        self._model.eval()

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a helpful assistant. Answer the question using the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def answer(self, question: str, context: str) -> Dict[str, Any]:
        if not self._model or not self._tokenizer:
            return self.mark_failed("Model not loaded yet.")

        prompt = self._build_prompt(question, context)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        )

        if self.config.device:
            inputs = {key: value.to(self.config.device) for key, value in inputs.items()}

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0.0,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=generation_config,
            )

        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return {
            "model": self.name,
            "answer": answer,
            "metadata": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens,
                "top_p": self.config.top_p,
            },
        }

    def sample_answers(
        self,
        question: str,
        context: str,
        num_samples: int = 4,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """Return multiple stochastic generations for self-consistency checks."""
        if not self._model or not self._tokenizer:
            return []

        prompt = self._build_prompt(question, context)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        )

        if self.config.device:
            inputs = {key: value.to(self.config.device) for key, value in inputs.items()}

        temp = temperature if temperature is not None else max(self.config.temperature, 0.6)
        tp = top_p if top_p is not None else self.config.top_p

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=temp,
            top_p=tp,
            do_sample=True,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=generation_config,
                num_return_sequences=num_samples,
            )

        return [
            self._tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]

