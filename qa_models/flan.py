"""FLAN-T5 based QA backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
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
        self._model: Optional[Any] = None  # Can be Seq2SeqLM or CausalLM
        self._is_causal: bool = False  # Track if this is a causal LM

    def load(self) -> None:
        """Load tokenizer and model weights."""
        logger.info("Loading model %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Detect model type: check if it's a causal LM or seq2seq
        config = AutoConfig.from_pretrained(self.config.model_name)
        if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
            # Encoder-decoder model (T5, FLAN-T5, etc.)
            logger.info("Detected encoder-decoder model, using AutoModelForSeq2SeqLM")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            self._is_causal = False
        else:
            # Decoder-only causal LM (TinyLlama, GPT, etc.)
            logger.info("Detected causal LM, using AutoModelForCausalLM")
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            self._is_causal = True
            # For causal LMs, we need to add padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        if self.config.device:
            logger.info("Moving model to device %s", self.config.device)
            self._model.to(self.config.device)

        self._model.eval()

    def _build_prompt(self, question: str, context: str) -> str:
        # For chat models, use a more structured format
        if self._is_causal:
            # Use a format that works better for causal LMs
            return (
                f"Context: {context}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
        else:
            # Original format for seq2seq models
            return (
                "You are a helpful assistant. Answer the question using the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )

    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """Deterministic baseline answer using greedy decoding (no sampling)."""
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

        # For the baseline, we want *greedy* decoding with no sampling.
        # To avoid warnings about temperature/top_p being ignored, we do NOT
        # set them here – only max_new_tokens + do_sample=False.
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id if self._is_causal else None,
        )

        # For causal LMs, add stop sequences to prevent over-generation
        stop_sequences = None
        if self._is_causal:
            # Stop on newlines or question markers that might indicate continuation
            stop_sequences = ["\n\nQuestion:", "\n\nContext:", "\n\nAnswer:"]
            stop_token_ids = []
            for seq in stop_sequences:
                tokens = self._tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.append(tokens[0])  # Use first token of each sequence
            if stop_token_ids:
                generation_config.eos_token_id = stop_token_ids[0] if stop_token_ids else self._tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # For causal LMs, we need to extract only the newly generated tokens
        if self._is_causal:
            # Remove the input prompt from the output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            # Stop at first newline or question marker to prevent continuation
            for stop_seq in ["\n\nQuestion:", "\n\nContext:", "\n\nAnswer:", "\n\n"]:
                if stop_seq in answer:
                    answer = answer.split(stop_seq)[0].strip()
                    break
        else:
            # For seq2seq models, the output is already just the generated part
            answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return {
            "model": self.name,
            "answer": answer,
            "metadata": {
                # Explicitly record that this was a greedy baseline run.
                "temperature": 0.0,
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
            pad_token_id=self._tokenizer.pad_token_id if self._is_causal else None,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=generation_config,
                num_return_sequences=num_samples,
            )

        # For causal LMs, extract only newly generated tokens
        if self._is_causal:
            input_length = inputs['input_ids'].shape[1]
            answers = []
            for output in outputs:
                generated_tokens = output[input_length:]
                answer = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                # Stop at first newline or question marker to prevent continuation
                for stop_seq in ["\n\nQuestion:", "\n\nContext:", "\n\nAnswer:", "\n\n"]:
                    if stop_seq in answer:
                        answer = answer.split(stop_seq)[0].strip()
                        break
                answers.append(answer)
            return answers
        else:
            return [
                self._tokenizer.decode(output, skip_special_tokens=True).strip()
                for output in outputs
            ]

