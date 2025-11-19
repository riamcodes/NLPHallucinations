"""
Checks whether the answer changes when the most supportive
sentence in the context is removed

If removing the evidence does not change the answer,
the model may be relying on parametric knowledge 
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from collections import Counter

from .base import DetectionResult, Detector

TOKEN_PATTERN = re.compile(r"\w+")


# Helper functions
def _tokenise(text: str) -> Counter:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return Counter(tokens)


def _split_sentences(text: str) -> List[str]:
    # Very simple sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _word_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-like token overlap ratio between two strings."""
    a_tokens = set(TOKEN_PATTERN.findall(a.lower()))
    b_tokens = set(TOKEN_PATTERN.findall(b.lower()))

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


# Detector Implementation
class ContextAblationDetector(Detector):
    """
    Removes the most supporting sentence from the context and checks how
    much the answer changes

    Process:
      1. Compute lexical support score for each sentence
      2. Remove the top-supporting sentence
      3. Regenerate answer under ablated context
      4. Compare new answer to original answer

    If model gives the SAME answer even after removing the evidence,
    label = "flag_context_ignored"
    """

    def __init__(self, support_threshold: float = 0.2, sensitivity_threshold: float = 0.3):
        super().__init__(name="context_ablation")
        self.support_threshold = support_threshold
        self.sensitivity_threshold = sensitivity_threshold

    def evaluate(self, result: Dict[str, Any], model: Any) -> DetectionResult:
        question = result.get("question", "")
        context = result.get("context", "")
        base_answer = result.get("answer", "")

        if not context.strip() or not base_answer.strip():
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "missing_context_or_answer"},
            )

        # 1. Split context into sentences
        sentences = _split_sentences(context)
        if not sentences:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "no_sentences_found"},
            )

        # 2. Find most-supporting sentence
        sentence_supports = []
        answer_tokens = _tokenise(base_answer)

        for s in sentences:
            ctx_tokens = _tokenise(s)
            if not ctx_tokens or not answer_tokens:
                support_ratio = 0.0
            else:
                overlap = sum(min(answer_tokens[t], ctx_tokens.get(t, 0)) for t in answer_tokens)
                answer_len = sum(answer_tokens.values()) or 1
                support_ratio = overlap / answer_len

            sentence_supports.append((s, support_ratio))

        # Select highest-support sentence
        support_sent, max_support = max(sentence_supports, key=lambda x: x[1])

        # 3. Remove this sentence
        ablated_context = " ".join(s for s in sentences if s != support_sent).strip()
        if not ablated_context:
            # Everything removed - insufficient context
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "ablated_context_empty"},
            )

        # 4. Generate ablated answer
        backend = getattr(model, "answer", None)
        if backend is None:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="unsupported",
                details={"reason": "model_missing_answer_method"},
            )

        raw_ablated = backend(question=question, context=ablated_context)

        # Handle both string and dict outputs
        if isinstance(raw_ablated, dict):
            ablated_answer = raw_ablated.get("answer", "")
        else:
            ablated_answer = raw_ablated if isinstance(raw_ablated, str) else str(raw_ablated)

        # 5. Compare baseline vs. ablated
        similarity = _word_overlap_ratio(base_answer, ablated_answer)
        sensitivity = 1 - similarity  # high = very sensitive to context removal

        # 6. Decision rule
        if max_support > self.support_threshold and sensitivity < self.sensitivity_threshold:
            label = "flag_context_ignored"
        else:
            label = "context_sensitive"

        return DetectionResult(
            detector=self.name,
            score=round(sensitivity, 4),
            label=label,
            details={
                "max_support": round(max_support, 4),
                "sensitivity": round(sensitivity, 4),
                "baseline_answer": base_answer,
                "ablated_answer": ablated_answer,
                "support_sentence": support_sent,
                "similarity": round(similarity, 4),
                "thresholds": {
                    "support_threshold": self.support_threshold,
                    "sensitivity_threshold": self.sensitivity_threshold,
                },
            },
        )
