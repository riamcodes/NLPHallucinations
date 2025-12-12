#!/usr/bin/env python3
"""Entry point for multi-model QA baselines used in hallucination experiments."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import torch

from detectors import (
    ContextOverlapDetector,
    SelfConsistencyDetector,
    ContextAblationDetector,
    TemperatureSensitivityDetector,
    SemanticOverlapDetector,
)
from experiment import QAExperiment
from qa_models import FlanConfig, FlanT5QA, LlamaCppConfig, LlamaCppQA, QAModel
from results import ResultLogger
from questions import load_question_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTIONS = load_question_set()

def detect_device() -> str | None:
    """Return the best available device for Transformers models."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


def build_models() -> List[QAModel]:
    """Construct the configured QA models."""
    models: List[QAModel] = []

    device = detect_device()
    models.append(
        FlanT5QA(
            name="flan-t5-base",
            config=FlanConfig(device=device),
        )
    )

    # Add TinyLlama using the same FlanT5QA interface
    # Using the chat version which is more commonly available
    models.append(
        FlanT5QA(
            name="tinyllama-1.1b",
            config=FlanConfig(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device=device,
            ),
        )
    )

    mistral_path = os.getenv("MISTRAL_GGUF_PATH")
    if mistral_path:
        models.append(
            LlamaCppQA(
                name="mistral-7b-instruct",
                config=LlamaCppConfig(model_path=Path(mistral_path)),
            )
        )
    else:
        logger.debug("MISTRAL_GGUF_PATH not set; skipping Mistral model.")

    llama3_path = os.getenv("LLAMA3_GGUF_PATH")
    if llama3_path:
        models.append(
            LlamaCppQA(
                name="llama-3-8b-instruct",
                config=LlamaCppConfig(model_path=Path(llama3_path)),
            )
        )
    else:
        logger.debug("LLAMA3_GGUF_PATH not set; skipping Llama-3 model.")


    return models


def display_results(results: List[dict]) -> None:
    """Pretty-print results to stdout."""
    print("=" * 80)
    print("Baseline Multi-Model QA Results")
    print("=" * 80)

    for item in results:
        qid = item.get("qid")
        category = item.get("category")
        question = item.get("question")

        # Header line – now includes qid + category
        print(f"[{item.get('model')}] {qid} ({category})")
        print(f"  Question: {question}")

        # Only print context for context-dependent questions
        if category == "context_dependent":
            context = item.get("context") or ""
            if context.strip():
                print(f"  Context: {context}")

        # Error handling
        if "error" in item:
            print(f"  Error: {item['error']}")
        else:
            # Answer
            print(f"  Answer: {item.get('answer')}")

            # Metadata (temperature, etc.)
            metadata = item.get("metadata", {})
            if metadata:
                print(f"  Metadata: {metadata}")

            # Detector outputs
            detections = item.get("detections", {})
            if detections:
                print("  Detections:")
                for name, payload in detections.items():
                    print(
                        f"    - {name}: {payload.get('label')} "
                        f"(score={payload.get('score')})"
                    )

        print("-" * 80)


def main() -> None:
    """Run the QA experiment."""
    models = build_models()
    if not models:
        logger.error("No models configured. Set the required environment variables and retry.")
        return

    detectors = [
        ContextOverlapDetector(),
        SelfConsistencyDetector(),
        ContextAblationDetector(),
        TemperatureSensitivityDetector(),
        SemanticOverlapDetector(),
    ]

    result_logger = ResultLogger()

    experiment = QAExperiment(models, detectors=detectors, logger_=result_logger)
    try:
        results = experiment.run(QUESTIONS)
    finally:
        result_logger.close()

    display_results(results)


if __name__ == "__main__":
    main()
