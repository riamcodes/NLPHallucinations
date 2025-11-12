"""Abstract base class for question answering models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QAModel(ABC):
    """Common interface every QA backend must implement."""

    def __init__(self, name: str):
        self.name = name
        self._loaded = False
        self._last_error: Optional[str] = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """Return an answer dictionary for the provided question/context pair."""

    def ensure_loaded(self) -> bool:
        """Ensure the backend is initialised exactly once."""
        if self._loaded:
            return True

        try:
            self.load()
            self._loaded = True
            self._last_error = None
        except Exception as exc:  # pragma: no cover - defensive logging
            self._last_error = str(exc)
            logger.exception("Failed to load model %s: %s", self.name, exc)

        return self._loaded

    @property
    def last_error(self) -> Optional[str]:
        """Return the latest load or inference error."""
        return self._last_error

    def mark_failed(self, message: str) -> Dict[str, Any]:
        """Utility for subclasses to return a structured failure."""
        logger.error("Model %s failed to answer: %s", self.name, message)
        return {"model": self.name, "error": message}

