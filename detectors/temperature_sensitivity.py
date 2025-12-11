"""Detector that measures answer stability across temperatures."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import DetectionResult, Detector

TOKEN_PATTERN = re.compile(r"\w+")


def _tokenise(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _word_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-like token overlap ratio between two strings."""
    a_tokens = set(_tokenise(a))
    b_tokens = set(_tokenise(b))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


class TemperatureSensitivityDetector(Detector):
    """
    Samples answers at multiple temperatures and measures how much
    they drift away from the baseline answer.

    Intuition:
    - If high-temperature answers are very different from the
      low-temperature baseline, the model is unstable and more likely
      to hallucinate.
    """

    def __init__(
        self,
        temperatures: list[float] | None = None,
        samples_per_temp: int = 2,
        mean_drift_threshold: float = 0.35,
        max_drift_threshold: float = 0.6,
    ) -> None:
        super().__init__(name="temperature_sensitivity")
        self.temperatures = temperatures or [0.3, 0.7, 1.0]
        self.samples_per_temp = samples_per_temp
        self.mean_drift_threshold = mean_drift_threshold
        self.max_drift_threshold = max_drift_threshold

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
        baseline_answer = result.get("answer", "").strip()

        if not baseline_answer:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "missing_baseline_answer"},
            )

        per_temp_similarities: dict[float, list[float]] = {}
        all_drifts: list[float] = []

        for temp in self.temperatures:
            replies = sampler(
                question=question,
                context=context,
                num_samples=self.samples_per_temp,
                temperature=max(temp, 1e-3),  # avoid exact 0.0 just in case
            )
            sims = []
            for r in replies:
                sim = _word_overlap_ratio(baseline_answer, r)
                sims.append(sim)
                all_drifts.append(1.0 - sim)
            per_temp_similarities[temp] = sims

        if not all_drifts:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "no_samples_generated"},
            )

        mean_drift = sum(all_drifts) / len(all_drifts)
        max_drift = max(all_drifts)

        # Decision rule: flag if answers change a lot with temperature
        is_flag = (mean_drift >= self.mean_drift_threshold) or (
            max_drift >= self.max_drift_threshold
        )
        label = "flag_temp_sensitive" if is_flag else "temperature_stable"

        return DetectionResult(
            detector=self.name,
            score=round(mean_drift, 4),
            label=label,
            details={
                "mean_drift": round(mean_drift, 4),
                "max_drift": round(max_drift, 4),
                "temperatures": self.temperatures,
                "samples_per_temp": self.samples_per_temp,
                "per_temp_similarities": {
                    str(t): [round(s, 4) for s in sims]
                    for t, sims in per_temp_similarities.items()
                },
                "thresholds": {
                    "mean_drift_threshold": self.mean_drift_threshold,
                    "max_drift_threshold": self.max_drift_threshold,
                },
            },
        )
