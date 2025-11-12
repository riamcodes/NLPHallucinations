"""Self-consistency detector using multiple stochastic samples."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import DetectionResult, Detector


def _unique_responses(responses: List[str]) -> List[str]:
    seen = set()
    uniques = []
    for response in responses:
        normalised = response.strip().lower()
        if normalised not in seen:
            seen.add(normalised)
            uniques.append(response)
    return uniques


class SelfConsistencyDetector(Detector):
    """Generates multiple samples and measures disagreement."""

    def __init__(self, samples: int = 4, temperature: float = 0.6):
        super().__init__(name="self_consistency")
        self.samples = samples
        self.temperature = temperature

    def evaluate(self, result: Dict[str, Any], model: Any) -> DetectionResult:
        sampler = getattr(model, "sample_answers", None)
        if sampler is None:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="unsupported",
                details={"reason": "sampler_not_available"},
            )

        question = result.get("question", "")
        context = result.get("context", "")

        replies = sampler(
            question=question,
            context=context,
            num_samples=self.samples,
            temperature=self.temperature,
        )

        unique_replies = _unique_responses(replies)
        diversity = len(unique_replies) / self.samples if self.samples else 0.0
        label = "flag" if diversity > 0.4 else "consistent"

        return DetectionResult(
            detector=self.name,
            score=round(diversity, 4),
            label=label,
            details={
                "unique_replies": unique_replies,
                "samples": self.samples,
                "temperature": self.temperature,
            },
        )

