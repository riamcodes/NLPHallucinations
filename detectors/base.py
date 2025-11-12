"""Base classes for hallucination detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DetectionResult:
    """Outcome returned by a detector."""

    detector: str
    score: float
    label: str
    details: Dict[str, Any]


class Detector(ABC):
    """Shared detector interface."""

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, result: Dict[str, Any], model: Any) -> DetectionResult:
        """Return a detection outcome for a QA result."""

