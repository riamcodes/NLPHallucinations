"""Orchestrates running multiple QA models over a set of questions."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from detectors import Detector
from qa_models import QAModel
from results import ResultLogger

logger = logging.getLogger(__name__)


class QAExperiment:
    """Runs a collection of QA models on shared question/context inputs."""

    def __init__(
        self,
        models: Sequence[QAModel],
        detectors: Sequence[Detector] | None = None,
        logger_: ResultLogger | None = None,
    ):
        self.models: List[QAModel] = list(models)
        self.detectors: List[Detector] = list(detectors) if detectors else []
        self.logger = logger_

    def run(self, questions: Iterable[dict]) -> List[dict]:
        """Execute the experiment and return aggregated results."""
        all_results: List[dict] = []

        for model in self.models:
            if not model.ensure_loaded():
                logger.warning("Skipping model %s: %s", model.name, model.last_error)
                continue

            for item in questions:
                question = item.get("question", "")
                context = item.get("context", "")
                result = model.answer(question=question, context=context)

                result.update(
                    {
                        "qid": item.get("qid"),
                        "category": item.get("category"),
                        "question": question,
                        "context": context,
                        # carry through the per-question "risk" label
                        "gold_is_hallucination": (item.get("gold_label") or {}).get(
                            "is_hallucination"
                        ),
                    }
                )

                detections = {}
                for detector in self.detectors:
                    detection = detector.evaluate(result, model)
                    detections[detection.detector] = {
                        "score": detection.score,
                        "label": detection.label,
                        "details": detection.details,
                    }
                if detections:
                    result["detections"] = detections

                if self.logger:
                    self.logger.log(result)

                all_results.append(result)

        return all_results
