"""Semantic similarity detector between answer and context."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import DetectionResult, Detector

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - import guard
    SentenceTransformer = None


class SemanticOverlapDetector(Detector):
    """
    Uses a sentence-transformers model to compute cosine similarity
    between the context and the answer.

    Low similarity suggests the answer is not semantically grounded
    in the context, even if some words overlap.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
    ) -> None:
        super().__init__(name="semantic_overlap")
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> Optional[SentenceTransformer]:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            return None
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def evaluate(self, result: Dict[str, Any], model: Any) -> DetectionResult:
        embedder = self._ensure_model()
        if embedder is None:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="unsupported",
                details={"reason": "sentence_transformers_not_available"},
            )

        context = result.get("context", "").strip()
        answer = result.get("answer", "").strip()

        if not context or not answer:
            return DetectionResult(
                detector=self.name,
                score=0.0,
                label="insufficient",
                details={"reason": "missing_context_or_answer"},
            )

        embeddings = embedder.encode(
            [context, answer], normalize_embeddings=True
        )
        ctx_vec, ans_vec = embeddings
        similarity = float((ctx_vec * ans_vec).sum())  # cosine because normalized

        label = "flag" if similarity < self.similarity_threshold else "support"

        return DetectionResult(
            detector=self.name,
            score=round(similarity, 4),
            label=label,
            details={
                "similarity": round(similarity, 4),
                "threshold": self.similarity_threshold,
                "model_name": self.model_name,
            },
        )
