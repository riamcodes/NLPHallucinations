"""Simple lexical-overlap detector comparing answer to context."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict

from .base import DetectionResult, Detector

TOKEN_PATTERN = re.compile(r"\w+")


def _tokenise(text: str) -> Counter:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return Counter(tokens)


class ContextOverlapDetector(Detector):
    """Flags answers with low lexical support in the supplied context."""

    def __init__(self, threshold: float = 0.2):
        super().__init__(name="context_overlap")
        self.threshold = threshold

    def evaluate(self, result: Dict[str, Any], model: Any) -> DetectionResult:
        context = result.get("context", "")
        answer = result.get("answer", "")

        context_counts = _tokenise(context)
        answer_counts = _tokenise(answer)
        if not context_counts or not answer_counts:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "missing_tokens"},
            )

        overlap = sum(
            min(answer_counts[token], context_counts.get(token, 0))
            for token in answer_counts
        )
        answer_len = sum(answer_counts.values())
        support_ratio = overlap / answer_len if answer_len else 0.0

        label = "flag" if support_ratio < self.threshold else "support"
        return DetectionResult(
            detector=self.name,
            score=round(support_ratio, 4),
            label=label,
            details={
                "answer_length": answer_len,
                "overlap_tokens": overlap,
                "threshold": self.threshold,
            },
        )

