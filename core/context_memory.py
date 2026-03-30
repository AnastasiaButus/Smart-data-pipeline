"""Persistent context memory helpers for pipeline steps."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger


class ContextMemory:
    """Read, write, and summarize compact pipeline context memory."""

    def __init__(self, path: str = "reports/context_memory.json") -> None:
        """Load an existing memory file or initialize an empty structure."""
        self._path = Path(path)
        self._data: dict[str, Any] = self._empty_payload()

        if not self._path.exists():
            logger.info("Context memory file does not exist yet: {}", self._path)
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = self._normalize_payload(raw)
            logger.info("Context memory loaded from {}", self._path)
        except Exception as exc:
            logger.warning(
                "Failed to load context memory from {}: {}. Using empty payload.",
                self._path,
                exc,
            )

    def load(self) -> dict[str, Any]:
        """Return the in-memory representation of the context store."""
        return deepcopy(self._data)

    def save(self) -> None:
        """Persist the current context payload to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Context memory saved to {}", self._path)

    def update(
        self,
        step: str,
        status: str,
        metrics: dict[str, Any],
        notes: str = "",
    ) -> None:
        """Update a step record and save the memory file immediately."""
        record = {
            "step": step,
            "status": status,
            "metrics": metrics,
            "notes": notes,
        }
        history = self._data.setdefault("history", {})
        history[step] = record
        self._data["step"] = step
        self._data["status"] = status
        self._data["metrics"] = metrics
        self._data["notes"] = notes
        self.save()

    def get(self, step: str | None = None) -> dict[str, Any]:
        """Return the full payload or a single step record if requested."""
        if step is None:
            return self.load()
        history = self._data.get("history", {})
        return deepcopy(history.get(step, {}))

    def get_summary_for_llm(self, max_chars: int = 500) -> str:
        """Return a compact summary of recent steps, capped by ``max_chars``."""
        history = self._data.get("history", {})
        if not history:
            return ""

        pieces: list[str] = []
        for step_key in self._sorted_steps(history.keys())[-3:]:
            record = history[step_key]
            metrics = record.get("metrics", {})
            total_rows = metrics.get("total_rows")
            source_distribution = metrics.get("source_distribution") or metrics.get(
                "sources", {}
            )
            class_count = metrics.get("recommended_classes_count")
            review_label = metrics.get("review_label")

            details: list[str] = []
            if total_rows is not None:
                details.append(f"rows={total_rows}")
            if source_distribution:
                top_sources = ", ".join(
                    f"{name}:{count}"
                    for name, count in list(source_distribution.items())[:3]
                )
                details.append(f"sources={top_sources}")
            if class_count is not None:
                details.append(f"classes={class_count}")
            if review_label:
                details.append(f"review={review_label}")

            notes = str(record.get("notes", "")).strip()
            piece = f"step {step_key} {record.get('status', '')}: {'; '.join(details)}"
            if notes:
                piece = f"{piece}; notes={notes}"
            pieces.append(piece.strip())

        summary = " | ".join(piece for piece in pieces if piece)
        if len(summary) > max_chars:
            return f"{summary[: max_chars - 3]}..."
        return summary

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        """Return the default empty payload shape."""
        return {
            "step": "",
            "status": "",
            "metrics": {},
            "notes": "",
            "history": {},
        }

    def _normalize_payload(self, raw: Any) -> dict[str, Any]:
        """Normalize legacy or partial JSON payloads to the current shape."""
        payload = self._empty_payload()

        if not isinstance(raw, dict):
            return payload

        if "history" in raw and isinstance(raw["history"], dict):
            payload.update(
                {
                    "step": str(raw.get("step", "")),
                    "status": str(raw.get("status", "")),
                    "metrics": raw.get("metrics", {}) or {},
                    "notes": str(raw.get("notes", "")),
                    "history": raw["history"],
                }
            )
            if not payload["step"] and payload["history"]:
                last_step = self._sorted_steps(payload["history"].keys())[-1]
                payload.update(payload["history"][last_step])
            return payload

        if {"step", "status", "metrics"}.issubset(raw.keys()):
            step = str(raw.get("step", ""))
            record = {
                "step": step,
                "status": str(raw.get("status", "")),
                "metrics": raw.get("metrics", {}) or {},
                "notes": str(raw.get("notes", "")),
            }
            payload.update(record)
            if step:
                payload["history"] = {step: record}
            return payload

        history_candidates = {
            key: value
            for key, value in raw.items()
            if isinstance(value, dict) and {"step", "status", "metrics"}.issubset(value)
        }
        if history_candidates:
            last_step = self._sorted_steps(history_candidates.keys())[-1]
            payload.update(history_candidates[last_step])
            payload["history"] = history_candidates
            return payload

        return payload

    @staticmethod
    def _sorted_steps(step_keys: Any) -> list[str]:
        """Sort step keys like ``1.2`` and ``1.10`` in numeric order."""
        return sorted(
            [str(step_key) for step_key in step_keys],
            key=lambda value: tuple(
                int(part) if part.isdigit() else part for part in value.split(".")
            ),
        )
