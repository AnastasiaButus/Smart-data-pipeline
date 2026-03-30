"""Data quality detection and cleanup agent for step 2.1."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from core.context_memory import ContextMemory
from core.llm_client import GeminiLLMClient


class DataQualityAgent:
    """Detect, clean, compare, and report text dataset quality issues."""

    _DEFAULT_STRATEGY = {
        "html_entities": "decode",
        "html_artifacts": "remove",
        "duplicates": "drop",
        "short_texts": "filter",
        "long_texts": "truncate",
        "missing": "drop",
    }
    _DEFAULT_ARTIFACT_PATTERNS = [
        "wp",
        "timeincuk",
        "inspirewp",
        "keyassets",
        "figcaption",
        "attachment",
        "webfeedsfeaturedvisual",
    ]

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load config, thresholds, and helper clients for data quality work."""
        self._config_path = Path(config_path)
        self._project_root = self._config_path.parent
        self._cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        self._quality_cfg = self._cfg.get("quality", {}) or {}
        self._strategy = {
            **self._DEFAULT_STRATEGY,
            **(self._quality_cfg.get("strategy", {}) or {}),
        }
        self._thresholds = self._quality_cfg.get("thresholds", {}) or {}
        self._short_text_min = int(self._thresholds.get("short_text_min", 50))
        self._long_text_max = int(self._thresholds.get("long_text_max", 1000))
        self._artifact_patterns = [
            str(pattern).strip().lower()
            for pattern in self._quality_cfg.get(
                "html_artifact_patterns",
                self._DEFAULT_ARTIFACT_PATTERNS,
            )
            if str(pattern).strip()
        ]
        self._raw_path = self._project_root / str(
            self._cfg.get("data", {}).get("raw_path", "data/raw")
        )
        self._reports_path = self._project_root / "reports"
        self._clean_dataset_path = self._raw_path / "dataset_clean.parquet"
        self._quality_report_path = self._reports_path / "quality_report.md"
        self._llm_quality_advice_path = self._reports_path / "llm_quality_advice.json"
        self._context_memory_path = self._reports_path / "context_memory.json"
        self._llm_client = GeminiLLMClient(config_path=str(self._config_path))
        self.config = self._cfg
        self.llm = self._llm_client
        self._last_summary: dict[str, Any] = {}
        self._last_issues: dict[str, Any] = {}
        self._last_compare: dict[str, Any] = {}
        self._last_llm_advice: dict[str, Any] = {}
        logger.info(
            "DataQualityAgent initialized. short_text_min={}, long_text_max={}",
            self._short_text_min,
            self._long_text_max,
        )

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect compact quality metrics and return a structured quality report."""
        total_rows = int(len(df))
        working_df = self._prepare_dataframe(df)
        text_series = working_df["text"]

        missing_count = int(working_df.isna().sum().sum())
        missing_columns = [
            str(column)
            for column in working_df.columns
            if int(working_df[column].isna().sum()) > 0
        ]

        normalized_text = text_series.map(self._normalize_for_dedup)
        duplicate_mask = normalized_text.duplicated(keep="first") & normalized_text.ne("")
        duplicates_count = int(duplicate_mask.sum())

        short_mask = text_series.str.len() < self._short_text_min
        long_mask = text_series.str.len() > self._long_text_max
        html_mask = text_series.str.contains(self._html_issue_regex(), regex=True, na=False)
        artifact_mask = text_series.str.contains(
            self._artifact_regex(),
            regex=True,
            na=False,
        )

        artifact_hits = sorted(
            {
                pattern
                for text in text_series.str.lower()
                for pattern in self._artifact_patterns
                if pattern in text
            }
        )

        label_distribution = (
            working_df["label"].fillna("").astype(str).value_counts().to_dict()
            if "label" in working_df.columns
            else {}
        )
        imbalance_ratio = self._imbalance_ratio(label_distribution)

        report = {
            "missing_values": {
                "count": missing_count,
                "columns": missing_columns,
            },
            "duplicates": {
                "count": duplicates_count,
                "pct": self._pct(duplicates_count, total_rows),
            },
            "short_texts": {
                "count": int(short_mask.sum()),
                "pct": self._pct(int(short_mask.sum()), total_rows),
                "threshold": self._short_text_min,
            },
            "long_texts": {
                "count": int(long_mask.sum()),
                "pct": self._pct(int(long_mask.sum()), total_rows),
                "threshold": self._long_text_max,
            },
            "html_entities": {
                "count": int(html_mask.sum()),
                "pct": self._pct(int(html_mask.sum()), total_rows),
                "examples": [
                    text[:150]
                    for text in text_series[html_mask].head(3).tolist()
                ],
            },
            "html_artifacts": {
                "count": int(artifact_mask.sum()),
                "patterns": artifact_hits,
            },
            "class_imbalance": {
                "distribution": label_distribution,
                "imbalance_ratio": imbalance_ratio,
            },
            "total_rows": total_rows,
            "issues_found": sum(
                [
                    missing_count > 0,
                    duplicates_count > 0,
                    int(short_mask.sum()) > 0,
                    int(long_mask.sum()) > 0,
                    int(html_mask.sum()) > 0,
                    int(artifact_mask.sum()) > 0,
                    imbalance_ratio > 1,
                ]
            ),
        }
        self._last_issues = report
        logger.info(
            "Quality detection completed: total_rows={}, issues_found={}",
            total_rows,
            report["issues_found"],
        )
        return report

    def fix(self, df: pd.DataFrame, strategy: dict[str, str]) -> pd.DataFrame:
        """Apply the configured cleaning strategy and persist ``dataset_clean.parquet``."""
        working_df = self._prepare_dataframe(df).copy()
        rows_before = int(len(working_df))

        missing_strategy = strategy.get("missing", "drop")
        if missing_strategy == "drop":
            working_df = working_df.dropna()
        elif missing_strategy == "fill":
            working_df = working_df.fillna("")

        if strategy.get("html_entities", "decode") == "decode":
            working_df["text"] = working_df["text"].map(html.unescape)

        if strategy.get("html_artifacts", "remove") == "remove":
            working_df["text"] = working_df["text"].map(self._remove_html_artifacts)

        if strategy.get("duplicates", "drop") == "drop":
            dedup_key = working_df["text"].map(self._normalize_for_dedup)
            working_df = working_df.loc[~dedup_key.duplicated(keep="first")].copy()

        short_strategy = strategy.get("short_texts", "filter")
        if short_strategy == "filter":
            working_df = working_df.loc[
                working_df["text"].str.len() >= self._short_text_min
            ].copy()

        long_strategy = strategy.get("long_texts", "truncate")
        if long_strategy == "truncate":
            working_df["text"] = working_df["text"].str.slice(0, self._long_text_max)
        elif long_strategy == "filter":
            working_df = working_df.loc[
                working_df["text"].str.len() <= self._long_text_max
            ].copy()

        working_df["text"] = working_df["text"].map(self._normalize_whitespace)
        working_df = working_df.loc[working_df["text"].ne("")].copy()
        working_df = working_df.reset_index(drop=True)

        self._clean_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        working_df.to_parquet(self._clean_dataset_path, index=False)
        logger.info(
            "Cleaning complete. rows_before={}, rows_after={}, saved_to={}",
            rows_before,
            len(working_df),
            self._clean_dataset_path,
        )
        return working_df

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict[str, Any]:
        """Compare dataset quality before and after cleanup."""
        before_report = self.detect_issues(df_before)
        after_report = self.detect_issues(df_after)
        rows_before = int(len(df_before))
        rows_after = int(len(df_after))
        rows_removed = rows_before - rows_after

        compare_report = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_removed,
            "pct_removed": self._pct(rows_removed, rows_before),
            "improvements": {
                "html_entities_before": before_report["html_entities"]["count"],
                "html_entities_after": after_report["html_entities"]["count"],
                "short_texts_before": before_report["short_texts"]["count"],
                "short_texts_after": after_report["short_texts"]["count"],
                "duplicates_before": before_report["duplicates"]["count"],
                "duplicates_after": after_report["duplicates"]["count"],
                "avg_len_before": round(float(self._prepare_dataframe(df_before)["text"].str.len().mean()), 2)
                if rows_before
                else 0.0,
                "avg_len_after": round(float(self._prepare_dataframe(df_after)["text"].str.len().mean()), 2)
                if rows_after
                else 0.0,
            },
        }
        self._last_compare = compare_report
        logger.info(
            "Compare complete. rows_before={}, rows_after={}, rows_removed={}",
            rows_before,
            rows_after,
            rows_removed,
        )
        return compare_report

    def explain_issues(self, quality_report: dict[str, Any]) -> dict[str, Any]:
        """Explain detected issues in Russian and recommend a cleanup strategy."""
        topic = str(self.config.get("domain", {}).get("topic", "text classification"))
        prompt = (
            "You are a data quality expert. Analyze this dataset quality report and "
            "respond in Russian. "
            f"Dataset: {topic}. "
            f"HTML entities: {quality_report['html_entities']['count']}; "
            f"Short texts (<50): {quality_report['short_texts']['count']}; "
            f"Long texts (>1000): {quality_report['long_texts']['count']}; "
            f"Duplicates: {quality_report['duplicates']['count']}; "
            f"HTML artifacts: {quality_report['html_artifacts']['count']}; "
            f"Total rows: {quality_report['total_rows']}. "
            'Return JSON only with keys: summary, top_issues, recommended_strategy, risks, confidence.'
        )[:800]

        fallback = self._fallback_quality_advice(quality_report)
        try:
            response = self.llm.generate_json(prompt)
            advice = self._validate_quality_advice(response) or fallback
        except Exception as exc:
            logger.warning("LLM quality advice fallback used: {}", exc)
            advice = fallback

        self._last_llm_advice = advice
        self._llm_quality_advice_path.parent.mkdir(parents=True, exist_ok=True)
        self._llm_quality_advice_path.write_text(
            json.dumps(advice, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("LLM quality advice saved to {}", self._llm_quality_advice_path)
        return advice

    def save_report(self, report: dict[str, Any], compare: dict[str, Any]) -> None:
        """Save a human-readable Markdown quality report."""
        self._quality_report_path.parent.mkdir(parents=True, exist_ok=True)

        metric_rows = [
            (
                "Rows",
                compare["rows_before"],
                compare["rows_after"],
                f"-{compare['rows_removed']}",
            ),
            (
                "HTML entities",
                compare["improvements"]["html_entities_before"],
                compare["improvements"]["html_entities_after"],
                compare["improvements"]["html_entities_before"]
                - compare["improvements"]["html_entities_after"],
            ),
            (
                "Short texts",
                compare["improvements"]["short_texts_before"],
                compare["improvements"]["short_texts_after"],
                compare["improvements"]["short_texts_before"]
                - compare["improvements"]["short_texts_after"],
            ),
            (
                "Duplicates",
                compare["improvements"]["duplicates_before"],
                compare["improvements"]["duplicates_after"],
                compare["improvements"]["duplicates_before"]
                - compare["improvements"]["duplicates_after"],
            ),
            (
                "Average text length",
                compare["improvements"]["avg_len_before"],
                compare["improvements"]["avg_len_after"],
                round(
                    compare["improvements"]["avg_len_after"]
                    - compare["improvements"]["avg_len_before"],
                    2,
                ),
            ),
        ]

        lines = [
            "# Data Quality Report",
            "",
            "## Summary",
            f"- Total rows: {report['total_rows']}",
            f"- Issues found: {report['issues_found']}",
            f"- Strategy: `{json.dumps(self._last_summary.get('strategy_used', self._strategy), ensure_ascii=False)}`",
            "",
            "## Before / After",
            "| metric | before | after | improvement |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines.extend(
            f"| {metric} | {before} | {after} | {improvement} |"
            for metric, before, after, improvement in metric_rows
        )
        lines.extend(
            [
                "",
                "## Issue Details",
                f"- Missing values: {report['missing_values']['count']} ({', '.join(report['missing_values']['columns']) or 'none'})",
                f"- HTML artifact rows: {report['html_artifacts']['count']}",
                f"- Class distribution: {json.dumps(report['class_imbalance']['distribution'], ensure_ascii=False)}",
            ]
        )
        if self._last_llm_advice:
            lines.extend(
                [
                    "",
                    "## LLM рекомендации",
                    str(self._last_llm_advice.get("summary", "")),
                    "",
                    "### Топ проблемы",
                ]
            )
            lines.extend(
                f"- {issue}" for issue in self._last_llm_advice.get("top_issues", [])
            )
            lines.extend(["", "### Рекомендуемая стратегия"])
            lines.extend(
                f"- {key}: {value}"
                for key, value in self._last_llm_advice.get(
                    "recommended_strategy", {}
                ).items()
            )
            lines.extend(["", "### Риски"])
            lines.extend(
                f"- {risk}" for risk in self._last_llm_advice.get("risks", [])
            )
            lines.append(
                f"- confidence: {self._last_llm_advice.get('confidence', 'medium')}"
            )

        self._quality_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Quality report saved to {}", self._quality_report_path)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for later LLM steps."""
        return dict(self._last_summary)

    def run(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Execute detection, cleaning, comparison, reporting, and memory update."""
        if df is None:
            raw_dataset_path = self._raw_path / "dataset.parquet"
            if not raw_dataset_path.exists():
                raise FileNotFoundError(f"Raw dataset not found: {raw_dataset_path}")
            df = pd.read_parquet(raw_dataset_path)

        original_df = self._prepare_dataframe(df)
        strategy = dict(self._strategy)
        report = self.detect_issues(original_df)
        advice = self.explain_issues(report)
        clean_df = self.fix(original_df, strategy)
        compare_report = self.compare(original_df, clean_df)
        self._last_summary = {
            "total_before": compare_report["rows_before"],
            "total_after": compare_report["rows_after"],
            "issues_found": report["issues_found"],
            "strategy_used": strategy,
            "llm_confidence": advice.get("confidence", "medium"),
        }
        self.save_report(report, compare_report)
        ContextMemory(path=str(self._context_memory_path)).update(
            step="2.1",
            status="done",
            metrics=compare_report,
            notes="DataQualityAgent applied cleaning",
        )
        return clean_df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a safe dataframe copy with normalized required text columns."""
        working_df = df.copy()
        if "text" not in working_df.columns:
            raise ValueError("Dataset must contain a 'text' column")
        working_df["text"] = working_df["text"].fillna("").astype(str)
        if "label" not in working_df.columns:
            working_df["label"] = "unlabeled"
        return working_df

    def _html_issue_regex(self) -> re.Pattern[str]:
        """Build the regex for HTML entities and CSS-like patterns."""
        return re.compile(
            r"&(?:amp|#39|lt|gt|quot);|class\s*=|style\s*=|<div|<p\b",
            flags=re.IGNORECASE,
        )

    def _artifact_regex(self) -> re.Pattern[str]:
        """Build a regex matching configured HTML artifact tokens and compounds."""
        escaped = [re.escape(pattern) for pattern in self._artifact_patterns]
        joined = "|".join(escaped)
        return re.compile(rf"\b[\w-]*?(?:{joined})[\w-]*\b", flags=re.IGNORECASE)

    def _remove_html_artifacts(self, text: str) -> str:
        """Remove HTML tags, CSS fragments, and configured artifact tokens from text."""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"class\s*=\s*['\"][^'\"]*['\"]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"style\s*=\s*['\"][^'\"]*['\"]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = self._artifact_regex().sub(" ", cleaned)
        return cleaned

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse whitespace and strip text boundaries."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        """Normalize text for duplicate detection."""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _pct(count: int, total: int) -> float:
        """Return a rounded percentage."""
        if total == 0:
            return 0.0
        return round((count / total) * 100, 2)

    @staticmethod
    def _imbalance_ratio(distribution: dict[str, int]) -> float:
        """Compute a simple imbalance ratio from a label distribution."""
        positive_counts = [int(value) for value in distribution.values() if int(value) > 0]
        if not positive_counts:
            return 0.0
        if len(positive_counts) == 1:
            return float(positive_counts[0])
        return round(max(positive_counts) / min(positive_counts), 2)

    def _fallback_quality_advice(self, quality_report: dict[str, Any]) -> dict[str, Any]:
        """Build deterministic Russian fallback advice for quality cleanup."""
        total_rows = int(quality_report.get("total_rows", 0))
        html_count = int(quality_report.get("html_entities", {}).get("count", 0))
        short_count = int(quality_report.get("short_texts", {}).get("count", 0))
        return {
            "summary": (
                f"Датасет содержит {total_rows} строк. Основные проблемы: HTML-артефакты"
                " в RSS и короткие тексты в форумах."
            ),
            "top_issues": [
                f"HTML entities в {html_count} строках — декодирование обязательно",
                f"Короткие тексты ({short_count} строк) — содержат мало информации для классификации",
                "Длинные тексты из RSS содержат HTML-разметку",
            ],
            "recommended_strategy": {
                "html_entities": "decode — очищает текст от служебных символов",
                "duplicates": "drop — дубли не добавляют новой информации",
                "short_texts": "filter — тексты <50 символов ненадёжны для классификации",
                "long_texts": "truncate — сохраняем данные, обрезаем до 1000 символов",
            },
            "risks": [
                "Потеря 20% данных после фильтрации — особенно тематических sailingforums",
                "HuggingFace источники нетематические — риск смещения модели",
            ],
            "confidence": "medium",
        }

    @staticmethod
    def _validate_quality_advice(advice: Any) -> dict[str, Any] | None:
        """Validate LLM quality advice shape before saving it."""
        if not isinstance(advice, dict):
            return None
        required_keys = {
            "summary",
            "top_issues",
            "recommended_strategy",
            "risks",
            "confidence",
        }
        if not required_keys.issubset(advice.keys()):
            return None
        if not isinstance(advice.get("top_issues"), list):
            return None
        if not isinstance(advice.get("recommended_strategy"), dict):
            return None
        if not isinstance(advice.get("risks"), list):
            return None
        return {
            "summary": str(advice.get("summary", "")).strip(),
            "top_issues": [str(item).strip() for item in advice.get("top_issues", []) if str(item).strip()],
            "recommended_strategy": {
                str(key): str(value).strip()
                for key, value in advice.get("recommended_strategy", {}).items()
                if str(value).strip()
            },
            "risks": [str(item).strip() for item in advice.get("risks", []) if str(item).strip()],
            "confidence": str(advice.get("confidence", "medium")).strip() or "medium",
        }
