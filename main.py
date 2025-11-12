#!/usr/bin/env python3
"""Entry point for multi-model QA baselines used in hallucination experiments."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import torch

from detectors import ContextOverlapDetector, SelfConsistencyDetector
from experiment import QAExperiment
from qa_models import FlanConfig, FlanT5QA, LlamaCppConfig, LlamaCppQA, QAModel
from results import ResultLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
    },
    {
        "question": "Who is Dr. David Lin?",
        "context": "Dr. Lin is a professor of computer science at Southern Methodist University, specializing in artificial intelligence research.",
    },
    {
        "question": "Which city hosts the annual Aurora Tech Expo?",
        "context": "The Aurora Tech Expo is a technology showcase that rotates between Chicago and Denver in alternating years. The 2025 event is scheduled for Chicago.",
    },
    {
        "question": "Did the mission to Europa succeed in collecting ice samples?",
        "context": "NASA's 2024 Europa Clipper mission performed multiple flybys of Europa to analyze the moon's ice shell. Technical issues prevented the lander from drilling, so no physical ice samples were collected.",
    },
    {
        "question": "Which antibiotic was named in the incident report?",
        "context": "The hospital's pharmacovigilance unit recorded three adverse reactions to the antibiotic amoxicillin in the incident report.",
    },
    {
        "question": "What year did the Meridian Accord take effect?",
        "context": "The Meridian Accord, a global cybersecurity treaty, was ratified in 2024 and took effect in January 2025.",
    },
    {
        "question": "Which ingredients were listed for the Solaris energy drink?",
        "context": "Solaris is a prototype energy drink made from green tea extract, guarana, and vitamin B12.",
    },
    {
        "question": "Where will the 2026 Coastal Resilience Summit be held?",
        "context": "The Coastal Resilience Summit alternates between Miami and New Orleans. The 2026 edition will be held in Miami.",
    },
    {
        "question": "Who authored the internal audit on Helios Microgrid?",
        "context": "The internal audit of the Helios Microgrid project was performed by Deloitte Consulting.",
    },
    {
        "question": "Which team won the 2024 Interstellar Robotics Challenge?",
        "context": "The 2024 Interstellar Robotics Challenge was won by Team NovaTech from MIT.",
    },
    {
        "question": "What was the outcome of the Apex semiconductor recall?",
        "context": "Apex Semiconductors issued a voluntary recall for its Aurora-9 chip after discovering a thermal defect. Replacement units were shipped to clients within six weeks.",
    },
    {
        "question": "Which fossil fuel subsidies did the Green Horizons bill remove?",
        "context": "The Green Horizons bill removed subsidies for coal mining and offshore oil drilling operations.",
    },
    {
        "question": "What malfunction caused the Skysail drone crash?",
        "context": "Investigators reported that the Skysail cargo drone crashed due to a software fault in the autopilot communication module.",
    },
    {
        "question": "Which astronaut led the Artemis Pathfinder mission?",
        "context": "Astronaut Jessica Watkins served as the mission commander for the Artemis Pathfinder mission.",
    },
    {
        "question": "What chemical triggered the lab evacuation at Ridgeview High?",
        "context": "Ridgeview High School evacuated its chemistry lab after an accidental spill of chlorine gas was detected by sensors.",
    },
    {
        "question": "Which country signed the OceanNet data sharing pact first?",
        "context": "Japan was the first country to sign the OceanNet data sharing pact in 2024.",
    },
    {
        "question": "Did Project Atlas achieve carbon-neutral operations?",
        "context": "Project Atlas achieved carbon-neutral operations in late 2024 by offsetting all remaining emissions through renewable energy credits.",
    },
    {
        "question": "What software vulnerability did the Nimbus patch fix?",
        "context": "Nimbus Cloud Services released a security patch fixing a cross-site scripting (XSS) vulnerability in its web dashboard.",
    },
    {
        "question": "Which exhibit replaced the Renaissance Wing at the city museum?",
        "context": "The city museum replaced the Renaissance Wing with a new exhibit titled 'Art and Innovation: The Age of Discovery.'",
    },
    {
        "question": "Who announced the OrionOS 5.0 launch date?",
        "context": "Orion Systems CEO Laura Chen announced the OrionOS 5.0 launch date during the 2025 developer conference keynote.",
    },
    {
        "question": "Which sponsor withdrew from the 2025 Global Health Forum?",
        "context": "PharmaLife Corporation withdrew from the 2025 Global Health Forum due to a restructuring of its public health division.",
    },
    {
        "question": "What was the final vote count on the Riverbend Initiative?",
        "context": "The Riverbend Initiative passed the city council with 8 votes in favor and 3 against, allocating funds to riverbank restoration.",
    },
    {
        "question": "Which language model powers the AtlasDocs summarizer?",
        "context": "AtlasDocs uses a fine-tuned FLAN-T5 language model for document summarization.",
    },
    {
        "question": "Did the QuantumLink satellite achieve stable orbit?",
        "context": "QuantumLink's first satellite successfully reached a stable low-Earth orbit after a brief correction burn.",
    },
    {
        "question": "Which foods triggered the allergen alert in cafeteria B?",
        "context": "Cafeteria B issued an allergen alert after a cross-contamination incident involving peanuts and shellfish.",
    },
]



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

    mistral_path = os.getenv("MISTRAL_GGUF_PATH")
    if mistral_path:
        models.append(
            LlamaCppQA(
                name="mistral-7b-instruct",
                config=LlamaCppConfig(model_path=Path(mistral_path)),
            )
        )
    else:
        logger.warning("Environment variable MISTRAL_GGUF_PATH not set; skipping Mistral model.")

    llama3_path = os.getenv("LLAMA3_GGUF_PATH")
    if llama3_path:
        models.append(
            LlamaCppQA(
                name="llama-3-8b-instruct",
                config=LlamaCppConfig(model_path=Path(llama3_path)),
            )
        )
    else:
        logger.warning("Environment variable LLAMA3_GGUF_PATH not set; skipping Llama-3 model.")

    return models


def display_results(results: List[dict]) -> None:
    """Pretty-print results to stdout."""
    print("=" * 80)
    print("Baseline Multi-Model QA Results")
    print("=" * 80)

    for item in results:
        print(f"[{item.get('model')}] Question: {item.get('question')}")
        if "error" in item:
            print(f"  Error: {item['error']}")
        else:
            print(f"  Answer: {item.get('answer')}")
            metadata = item.get("metadata", {})
            if metadata:
                print(f"  Metadata: {metadata}")
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
        ContextOverlapDetector(threshold=0.25),
        SelfConsistencyDetector(samples=5, temperature=0.6),
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
