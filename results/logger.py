"""Persist experiment outputs for later analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class ResultLogger:
    """Writes QA outputs and detector scores to a JSONL file."""

    def __init__(self, output_dir: Path | str = "runs", filename: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            filename = f"run-{timestamp}.jsonl"

        self.path = self.output_dir / filename
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        """Append a record to the log file."""
        json_record = json.dumps(record, ensure_ascii=False)
        self._file.write(json_record + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        if getattr(self, "_file", None):
            self._file.close()

