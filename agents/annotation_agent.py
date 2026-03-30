"""Annotation agent for zero-shot labeling and HITL queue generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from core.context_memory import ContextMemory
from core.llm_client import GeminiLLMClient


class AnnotationAgent:
    """Auto-label texts, score confidence, and prepare HITL review artifacts."""

    _DEFAULT_THRESHOLD = 0.7
    _DEFAULT_MODEL = "facebook/bart-large-mnli"
    _DEFAULT_BATCH_SIZE = 16
    _DEFAULT_LABEL_SOURCE = "zero_shot"

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load config, domain labels, and helper clients without loading the model."""
        self._config_path = Path(config_path)
        self._project_root = self._config_path.parent
        self._cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        self._annotation_cfg = self._cfg.get("annotation", {}) or {}
        self._domain_cfg = self._cfg.get("domain", {}) or {}
        self._data_cfg = self._cfg.get("data", {}) or {}

        self.classes = [
            str(label).strip()
            for label in self._domain_cfg.get("classes", [])
            if str(label).strip()
        ]
        self.review_label = str(
            self._domain_cfg.get("review_label", "other_or_offtopic")
        ).strip() or "other_or_offtopic"
        self._confidence_threshold = float(
            self._annotation_cfg.get(
                "confidence_threshold",
                self._DEFAULT_THRESHOLD,
            )
        )
        self._model_name = str(
            self._annotation_cfg.get("model", self._DEFAULT_MODEL)
        ).strip() or self._DEFAULT_MODEL
        self._batch_size = int(
            self._annotation_cfg.get("batch_size", self._DEFAULT_BATCH_SIZE)
        )
        self._label_source = str(
            self._annotation_cfg.get("label_source", self._DEFAULT_LABEL_SOURCE)
        ).strip() or self._DEFAULT_LABEL_SOURCE

        raw_path = str(self._data_cfg.get("raw_path", "data/raw"))
        labeled_path = str(self._data_cfg.get("labeled_path", "data/labeled"))
        review_queue_path = str(
            self._data_cfg.get("review_queue_path", "data/review_queue.csv")
        )
        self._clean_dataset_path = self._project_root / raw_path / "dataset_clean.parquet"
        self._annotated_dataset_path = self._project_root / labeled_path / "annotated.parquet"
        self._review_queue_path = self._project_root / review_queue_path
        self._annotation_spec_path = self._project_root / "reports" / "annotation_spec.md"
        self._labelstudio_path = (
            self._project_root / "reports" / "labelstudio_import.json"
        )
        self._context_memory_path = self._project_root / "reports" / "context_memory.json"

        self._pipeline: Any | None = None
        self._llm_client = GeminiLLMClient(config_path=str(self._config_path))
        self._last_summary: dict[str, Any] = {}
        self._last_quality: dict[str, Any] = {}
        logger.info(
            "AnnotationAgent initialized. classes={}, threshold={}, batch_size={}",
            len(self.classes),
            self._confidence_threshold,
            self._batch_size,
        )

    def _get_pipeline(self) -> Any | None:
        """Lazily load the zero-shot pipeline and return it when available."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline

            logger.info("Loading zero-shot model...")
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
            )
        except Exception as exc:
            logger.error("Failed to load zero-shot pipeline: {}", exc)
            self._pipeline = None
        return self._pipeline

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label each text row with zero-shot classification or fallback values."""
        working_df = self._prepare_dataframe(df)
        candidate_labels = self._candidate_labels()
        classifier = self._get_pipeline()

        if classifier is None:
            logger.warning("Zero-shot pipeline unavailable. Using annotation fallback.")
            working_df["label"] = self.review_label
            working_df["confidence"] = 0.0
            working_df["label_source"] = "fallback"
            return working_df

        labels: list[str] = []
        confidences: list[float] = []
        sources: list[str] = []
        processed = 0

        for start in range(0, len(working_df), self._batch_size):
            stop = start + self._batch_size
            batch = working_df.iloc[start:stop]
            texts = batch["text"].fillna("").astype(str).tolist()

            try:
                results = classifier(
                    texts,
                    candidate_labels=candidate_labels,
                    multi_label=False,
                )
                if isinstance(results, dict):
                    results = [results]
            except Exception as exc:
                logger.error(
                    "Zero-shot inference failed for rows {}-{}: {}. Falling back for batch.",
                    start,
                    min(stop, len(working_df)),
                    exc,
                )
                results = [
                    {"labels": [self.review_label], "scores": [0.0]}
                    for _ in texts
                ]
                batch_source = "fallback"
            else:
                batch_source = self._label_source

            for result in results:
                batch_labels = result.get("labels", []) if isinstance(result, dict) else []
                batch_scores = result.get("scores", []) if isinstance(result, dict) else []
                predicted_label = (
                    str(batch_labels[0]).strip() if batch_labels else self.review_label
                )
                confidence = float(batch_scores[0]) if batch_scores else 0.0
                labels.append(predicted_label or self.review_label)
                confidences.append(round(confidence, 6))
                sources.append(batch_source)

            processed += len(texts)
            if processed % 50 == 0 or processed == len(working_df):
                logger.info("Auto-label progress: {}/{}", processed, len(working_df))

        working_df["label"] = labels
        working_df["confidence"] = confidences
        working_df["label_source"] = sources
        return working_df

    def generate_spec(self, task: str | None = None) -> str:
        """Generate and save annotation instructions in Markdown."""
        task_name = str(task or self._domain_cfg.get("topic", "text classification"))
        prompt = (
            "Generate annotation specification in Russian for task: "
            f"{task_name}. Classes: {', '.join(self._candidate_labels())}. "
            "Return Markdown with sections: ## Задача, ## Классы (с определениями), "
            "## Примеры (3+ на класс), ## Граничные случаи. Max 600 chars."
        )[:600]

        try:
            response = self._llm_client.generate(prompt).strip()
        except Exception as exc:
            logger.warning("Gemini annotation spec fallback used: {}", exc)
            response = ""

        if not response or response == "{}" or "##" not in response:
            response = self._fallback_spec(task_name)

        self._annotation_spec_path.parent.mkdir(parents=True, exist_ok=True)
        self._annotation_spec_path.write_text(response, encoding="utf-8")
        logger.info("Annotation spec saved to {}", self._annotation_spec_path)
        return response

    def check_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute compact quality metrics for the current annotation output."""
        working_df = self._prepare_labeled_dataframe(df)
        total_labeled = int(len(working_df))
        confidence_series = working_df["confidence"].astype(float)
        low_conf_mask = confidence_series < self._confidence_threshold
        high_conf_mask = confidence_series >= self._confidence_threshold

        metrics = {
            "total_labeled": total_labeled,
            "label_distribution": (
                working_df["label"].astype(str).value_counts().to_dict()
            ),
            "confidence_mean": round(float(confidence_series.mean()), 4)
            if total_labeled
            else 0.0,
            "confidence_std": round(float(confidence_series.std(ddof=0)), 4)
            if total_labeled
            else 0.0,
            "low_confidence_count": int(low_conf_mask.sum()),
            "low_confidence_pct": self._pct(int(low_conf_mask.sum()), total_labeled),
            "high_confidence_count": int(high_conf_mask.sum()),
            "review_label_count": int(
                working_df["label"].astype(str).eq(self.review_label).sum()
            ),
            "label_source_distribution": (
                working_df["label_source"].astype(str).value_counts().to_dict()
            ),
        }
        self._last_quality = metrics
        return metrics

    def flag_for_review(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split labeled rows into confident and review subsets and persist the queue."""
        working_df = self._prepare_labeled_dataframe(df)
        confident_mask = working_df["confidence"].astype(float) >= self._confidence_threshold
        df_confident = working_df.loc[confident_mask].reset_index(drop=True)
        df_review = working_df.loc[~confident_mask].reset_index(drop=True)

        review_queue = pd.DataFrame(
            {
                "id": df_review["id"],
                "text": df_review["text"],
                "label": df_review["label"],
                "confidence": df_review["confidence"],
                "source": df_review["source"],
                "suggested_label": df_review["label"],
                "corrected_label": "",
            }
        )
        self._review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        review_queue.to_csv(self._review_queue_path, index=False, encoding="utf-8")
        logger.info("Review queue saved with {} rows to {}", len(review_queue), self._review_queue_path)
        return df_confident, df_review

    def export_to_labelstudio(self, df: pd.DataFrame) -> None:
        """Export labeled rows in Label Studio import format."""
        working_df = self._prepare_labeled_dataframe(df)
        payload: list[dict[str, Any]] = []
        for row in working_df.to_dict(orient="records"):
            payload.append(
                {
                    "id": row["id"],
                    "data": {"text": row["text"]},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "type": "choices",
                                    "value": {"choices": [row["label"]]},
                                    "from_name": "label",
                                    "to_name": "text",
                                }
                            ]
                        }
                    ],
                }
            )

        self._labelstudio_path.parent.mkdir(parents=True, exist_ok=True)
        self._labelstudio_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Label Studio import saved to {}", self._labelstudio_path)

    def summary(self) -> dict[str, Any]:
        """Return a compact annotation summary without raw texts."""
        return dict(self._last_summary)

    def run(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Run auto-labeling, quality checks, review split, and export artifacts."""
        if df is None:
            if not self._clean_dataset_path.exists():
                raise FileNotFoundError(
                    f"Clean dataset not found: {self._clean_dataset_path}"
                )
            df = pd.read_parquet(self._clean_dataset_path)

        labeled_df = self.auto_label(df)
        quality_metrics = self.check_quality(labeled_df)
        df_confident, _df_review = self.flag_for_review(labeled_df)
        self.generate_spec()
        self.export_to_labelstudio(labeled_df)

        self._annotated_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_parquet(self._annotated_dataset_path, index=False)
        logger.info("Annotated parquet saved to {}", self._annotated_dataset_path)

        self._last_summary = {
            "total": int(len(labeled_df)),
            "labeled": quality_metrics.get("total_labeled", 0),
            "confidence_mean": quality_metrics.get("confidence_mean", 0.0),
            "low_conf_pct": quality_metrics.get("low_confidence_pct", 0.0),
            "classes": self._candidate_labels(),
        }

        ContextMemory(path=str(self._context_memory_path)).update(
            step="3.1",
            status="done",
            metrics=quality_metrics,
            notes="AnnotationAgent auto-labeled",
        )
        return df_confident

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a dataframe for annotation while preserving existing columns."""
        working_df = df.copy()
        if "text" not in working_df.columns:
            raise ValueError("Dataset must contain a 'text' column")
        working_df["text"] = working_df["text"].fillna("").astype(str)
        if "id" not in working_df.columns:
            working_df["id"] = [str(index + 1) for index in range(len(working_df))]
        else:
            working_df["id"] = working_df["id"].fillna("").astype(str)
        if "source" not in working_df.columns:
            working_df["source"] = "unknown"
        else:
            working_df["source"] = working_df["source"].fillna("unknown").astype(str)
        return working_df

    def _prepare_labeled_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure labeled dataframe has all required annotation columns."""
        working_df = self._prepare_dataframe(df)
        if "label" not in working_df.columns:
            working_df["label"] = self.review_label
        else:
            working_df["label"] = working_df["label"].fillna(self.review_label).astype(str)
        if "confidence" not in working_df.columns:
            working_df["confidence"] = 0.0
        else:
            working_df["confidence"] = pd.to_numeric(
                working_df["confidence"],
                errors="coerce",
            ).fillna(0.0)
        if "label_source" not in working_df.columns:
            working_df["label_source"] = "fallback"
        else:
            working_df["label_source"] = working_df["label_source"].fillna("fallback").astype(str)
        return working_df

    def _candidate_labels(self) -> list[str]:
        """Return unique candidate labels including the review class."""
        ordered = [*self.classes, self.review_label]
        unique: list[str] = []
        seen: set[str] = set()
        for label in ordered:
            normalized = str(label).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _fallback_spec(self, task_name: str) -> str:
        """Build a compact deterministic Markdown spec in Russian."""
        class_lines = "\n".join(
            f"- `{label}`: тексты, где доминирует тема `{label}`."
            for label in self.classes
        )
        examples = "\n".join(
            f"- `{label}`: 3+ примера нужно собирать из типичных текстов по теме `{task_name}`."
            for label in self.classes[:3]
        )
        return (
            "## Задача\n"
            f"Размечать тексты по теме `{task_name}` и отправлять неясные или оффтопные примеры в `{self.review_label}`.\n\n"
            "## Классы (с определениями)\n"
            f"{class_lines}\n"
            f"- `{self.review_label}`: шум, оффтоп и неоднозначные случаи.\n\n"
            "## Примеры (3+ на класс)\n"
            f"{examples}\n"
            f"- `{self.review_label}`: эмоциональные посты, общий chatter, тексты без явной доменной связи.\n\n"
            "## Граничные случаи\n"
            "Если текст короткий, смешанный по темам или уверенность модели низкая, его нужно отправить в review_queue."
        )

    @staticmethod
    def _pct(count: int, total: int) -> float:
        """Return a rounded percentage value."""
        if total == 0:
            return 0.0
        return round((count / total) * 100, 2)
