"""Gemini-based domain reformulation utilities for step 1.3."""

from __future__ import annotations

import importlib
import json
import os
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

from core.context_memory import ContextMemory


class GeminiLLMClient:
    """Build compact dataset summaries and generate a domain spec with Gemini."""

    _DEFAULT_CLASSES = [
        "navigation",
        "safety",
        "equipment",
        "weather",
        "licensing",
    ]
    _DEFAULT_CLASS_KEYWORDS = {
        "navigation": [
            "route planning",
            "waypoint",
            "chartplotter",
            "tide",
            "course",
        ],
        "safety": [
            "life jacket",
            "colregs",
            "man overboard",
            "distress",
            "watchkeeping",
        ],
        "equipment": [
            "rigging",
            "anchor",
            "engine",
            "electronics",
            "maintenance",
        ],
        "weather": [
            "forecast",
            "wind",
            "storm",
            "barometer",
            "sea state",
        ],
        "licensing": [
            "regulation",
            "certificate",
            "compliance",
            "exam",
            "radio license",
        ],
        "seamanship": [
            "anchoring",
            "reefing",
            "berthing",
            "watch schedule",
            "passage planning",
        ],
        "maintenance": [
            "hull",
            "diesel",
            "battery",
            "repair",
            "inspection",
        ],
    }
    _STOPWORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "boat",
        "boats",
        "sail",
        "sailing",
        "yacht",
        "yachts",
        "sea",
        "also",
        "get",
        "got",
        "one",
        "two",
        "using",
        "used",
    }

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load configuration and environment variables without network access."""
        self._config_path = Path(config_path)
        self._project_root = self._config_path.parent
        self._cfg: dict[str, Any] = {}
        self._config_valid = False
        self._sdk_available = False
        self._client: Any | None = None

        load_dotenv(self._project_root / ".env")
        self._api_key = os.getenv("GEMINI_API_KEY", "").strip()

        try:
            self._cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
            self._config_valid = isinstance(self._cfg, dict) and bool(self._cfg.get("llm"))
        except Exception as exc:
            logger.error("Failed to read config from {}: {}", self._config_path, exc)
            self._cfg = {}

        self._domain_cfg = self._cfg.get("domain", {}) if self._config_valid else {}
        self._llm_cfg = self._cfg.get("llm", {}) if self._config_valid else {}
        self._model_name = str(self._llm_cfg.get("model", "gemini-1.5-flash"))
        self._temperature = float(self._llm_cfg.get("temperature", 0.2))
        self._max_tokens = int(self._llm_cfg.get("max_tokens", 800))
        self._max_prompt_chars = int(self._llm_cfg.get("max_prompt_chars", 2000))

        try:
            importlib.import_module("google.genai")
            self._sdk_available = True
        except Exception as exc:
            logger.warning("Gemini SDK import unavailable: {}", exc)

        logger.info(
            "GeminiLLMClient initialized. config_valid={}, sdk_available={}, api_key_present={}",
            self._config_valid,
            self._sdk_available,
            bool(self._api_key),
        )

    def is_available(self) -> bool:
        """Return ``True`` when config, SDK, and API key are all available."""
        return bool(self._api_key and self._config_valid and self._sdk_available)

    def build_dataset_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Build a compact summary for LLM use without raw text rows."""
        if "text" not in df.columns:
            raise ValueError("Dataset must contain a 'text' column")

        text_series = df["text"].fillna("").astype(str)
        text_lengths = text_series.str.len()
        tokens_per_row = [self._tokenize(text) for text in text_series]
        global_counter = Counter(
            token for row_tokens in tokens_per_row for token in row_tokens
        )

        source_distribution = (
            df["source"].fillna("unknown").astype(str).value_counts().to_dict()
            if "source" in df.columns
            else {}
        )

        top_keywords_by_source: dict[str, list[str]] = {}
        if "source" in df.columns:
            source_series = df["source"].fillna("unknown").astype(str)
            for source_name in source_series.unique():
                source_tokens = [
                    token
                    for text, source in zip(text_series, source_series)
                    if source == source_name
                    for token in self._tokenize(text)
                ]
                top_keywords_by_source[source_name] = [
                    token for token, _ in Counter(source_tokens).most_common(8)
                ]

        summary = {
            "total_rows": int(len(df)),
            "columns": [str(column) for column in df.columns.tolist()],
            "source_distribution": source_distribution,
            "avg_text_len": round(float(text_lengths.mean()), 2) if len(df) else 0.0,
            "min_text_len": int(text_lengths.min()) if len(df) else 0,
            "max_text_len": int(text_lengths.max()) if len(df) else 0,
            "pct_short_texts": round(float((text_lengths < 50).mean() * 100), 2)
            if len(df)
            else 0.0,
            "pct_long_texts": round(float((text_lengths > 500).mean() * 100), 2)
            if len(df)
            else 0.0,
            "html_entity_count": int(
                text_series.str.count(r"&(?:[a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);").sum()
            ),
            "top_keywords_global": [
                token for token, _ in global_counter.most_common(15)
            ],
            "top_keywords_by_source": top_keywords_by_source,
            "current_topic": str(
                self._domain_cfg.get("topic", "sailing and yacht navigation")
            ),
            "current_classes": self._safe_classes(
                self._domain_cfg.get("classes", self._DEFAULT_CLASSES)
            ),
        }
        logger.info(
            "Dataset summary built for {} rows and {} sources",
            summary["total_rows"],
            len(summary["source_distribution"]),
        )
        return summary

    def generate_domain_spec(
        self,
        topic: str,
        dataset_summary: dict[str, Any],
        current_classes: list[str],
    ) -> dict[str, Any]:
        """Generate a validated domain specification or fall back heuristically."""
        fallback = self.heuristic_fallback(topic, current_classes, dataset_summary)
        prompt = self._build_prompt(topic, dataset_summary, current_classes)

        if not self.is_available():
            logger.warning("Gemini is not available. Using heuristic fallback.")
            return fallback

        try:
            raw_spec = self._generate_json_response(prompt)
            validated = self._validate_spec(raw_spec)
            if not validated:
                logger.warning("Gemini returned invalid or empty JSON. Using fallback.")
                return fallback
            logger.info("Domain specification generated with Gemini")
            return validated
        except Exception as exc:
            logger.error("Gemini generation failed: {}. Using fallback.", exc)
            return fallback

    def heuristic_fallback(
        self,
        topic: str,
        current_classes: list[str],
        dataset_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a deterministic domain specification without any API calls."""
        classes = self._safe_classes(current_classes or dataset_summary.get("current_classes"))
        if not (5 <= len(classes) <= 7):
            classes = list(self._DEFAULT_CLASSES)

        source_fit: dict[str, dict[str, str]] = {}
        for source_name in dataset_summary.get("source_distribution", {}).keys():
            if source_name.startswith("huggingface_"):
                fit = "low"
                notes = (
                    "Useful for volume, but the source is partially off-topic and may bias"
                    " the task toward generic sentiment or emotion language."
                )
            elif source_name.startswith("stackexchange_"):
                fit = "high"
                notes = (
                    "Practical question data is close to sailing problem solving and future"
                    " annotation use cases."
                )
            elif source_name.startswith("rss_") or source_name == "sailingforums":
                fit = "high"
                notes = (
                    "Domain-relevant editorial or forum content is a strong fit for sailing,"
                    " navigation, seamanship, and safety topics."
                )
            else:
                fit = "medium"
                notes = "Source relevance is uncertain and should be reviewed during HITL."
            source_fit[source_name] = {"fit": fit, "notes": notes}

        keywords_by_class = {
            class_name: self._keywords_for_class(class_name, dataset_summary)
            for class_name in classes
        }
        collection_queries = {
            class_name: [
                f"{topic} {class_name}",
                f"{class_name} sailing best practices",
            ]
            for class_name in classes
        }

        return {
            "normalized_topic": topic,
            "recommended_classes": classes,
            "keywords_by_class": keywords_by_class,
            "collection_queries": collection_queries,
            "source_fit": source_fit,
            "review_label": "other_or_offtopic",
            "annotation_guidelines": [
                "Assign a class only when the text is clearly about sailing, yacht handling, navigation, safety, weather, licensing, or onboard equipment.",
                "Route noisy emotion-style or generic social posts to other_or_offtopic when they do not contain domain evidence.",
                "Prefer the operational topic discussed in the text over tone, sentiment, or writing style.",
            ],
            "risks": [
                "HuggingFace sources add useful volume but contain off-topic emotion and tweet content.",
                "Short RSS headlines may be ambiguous and can require review during annotation.",
                "Some topics overlap across safety, navigation, and equipment and need clear guidelines.",
            ],
            "llm_notes": (
                "Fallback domain spec was generated without Gemini. The dataset remains centered"
                " on sailing, while off-topic HuggingFace rows should be filtered by"
                " other_or_offtopic during annotation."
            ),
        }

    def save_domain_spec(
        self,
        spec: dict[str, Any],
        output_path: str = "reports/domain_spec.json",
    ) -> None:
        """Save the domain specification JSON artifact to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Domain spec saved to {}", path)

    def save_markdown_report(
        self,
        spec: dict[str, Any],
        summary: dict[str, Any],
        output_path: str = "reports/domain_reformulation.md",
    ) -> None:
        """Save a human-readable Markdown report for the reformulation step."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "# Domain Reformulation Report",
            "",
            "## Original Topic",
            str(summary.get("current_topic", "")),
            "",
            "## Dataset Summary",
            f"- total_rows: {summary.get('total_rows', 0)}",
            f"- columns: {', '.join(summary.get('columns', []))}",
            f"- source_distribution: {json.dumps(summary.get('source_distribution', {}), ensure_ascii=False)}",
            f"- avg_text_len: {summary.get('avg_text_len', 0)}",
            f"- min_text_len: {summary.get('min_text_len', 0)}",
            f"- max_text_len: {summary.get('max_text_len', 0)}",
            f"- pct_short_texts: {summary.get('pct_short_texts', 0)}",
            f"- pct_long_texts: {summary.get('pct_long_texts', 0)}",
            f"- html_entity_count: {summary.get('html_entity_count', 0)}",
            f"- top_keywords_global: {', '.join(summary.get('top_keywords_global', []))}",
            "",
            "## Recommended Classes",
        ]
        lines.extend(
            f"- {class_name}" for class_name in spec.get("recommended_classes", [])
        )
        lines.extend(["", "## Keywords By Class"])
        for class_name, keywords in spec.get("keywords_by_class", {}).items():
            lines.append(f"- {class_name}: {', '.join(keywords)}")

        lines.extend(["", "## Collection Queries"])
        for class_name, queries in spec.get("collection_queries", {}).items():
            lines.append(f"- {class_name}: {' | '.join(queries)}")

        lines.extend(["", "## Source Fit"])
        for source_name, fit_info in spec.get("source_fit", {}).items():
            lines.append(
                f"- {source_name}: {fit_info.get('fit', '')} - {fit_info.get('notes', '')}"
            )

        lines.extend(["", "## Risks"])
        lines.extend(f"- {risk}" for risk in spec.get("risks", []))
        lines.extend(
            [
                "",
                "## LLM Notes",
                str(spec.get("llm_notes", "")),
            ]
        )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Markdown report saved to {}", path)

    def update_config_domain(self, spec: dict[str, Any]) -> None:
        """Update the ``domain`` and ``llm`` sections in ``config.yaml`` safely."""
        config = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        domain = config.setdefault("domain", {})
        llm = config.setdefault("llm", {})

        domain["topic"] = domain.get("topic", self._domain_cfg.get("topic", ""))
        domain["normalized_topic"] = spec.get(
            "normalized_topic",
            domain.get("normalized_topic", ""),
        )
        domain["classes"] = spec.get(
            "recommended_classes",
            domain.get("classes", self._DEFAULT_CLASSES),
        )
        domain["keywords_by_class"] = spec.get("keywords_by_class", {})
        domain["review_label"] = spec.get("review_label", "other_or_offtopic")

        if "language" not in domain:
            domain["language"] = self._domain_cfg.get("language", "en")

        llm.setdefault("max_prompt_chars", 2000)

        self._cfg = config
        self._domain_cfg = domain
        self._llm_cfg = llm
        self._max_prompt_chars = int(llm.get("max_prompt_chars", 2000))

        self._config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        logger.info("Config updated with new domain specification at {}", self._config_path)

    def save_context_memory(
        self,
        spec: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        """Update ``reports/context_memory.json`` for step 1.3."""
        metrics = {
            "total_rows": int(summary.get("total_rows", 0)),
            "source_distribution": summary.get("source_distribution", {}),
            "recommended_classes_count": len(spec.get("recommended_classes", [])),
            "review_label": spec.get("review_label", "other_or_offtopic"),
        }
        ContextMemory(self._project_root / "reports" / "context_memory.json").update(
            step="1.3",
            status="done",
            metrics=metrics,
            notes="Domain spec generated with Gemini",
        )

    def run_from_dataset(
        self,
        dataset_path: str = "data/raw/dataset.parquet",
    ) -> dict[str, Any]:
        """Run the full step 1.3 pipeline from an existing parquet dataset."""
        path = Path(dataset_path)
        df = pd.read_parquet(path)
        summary = self.build_dataset_summary(df)
        topic = str(summary.get("current_topic", self._domain_cfg.get("topic", "")))
        current_classes = self._safe_classes(summary.get("current_classes", []))
        spec = self.generate_domain_spec(topic, summary, current_classes)

        self.save_domain_spec(spec, self._project_root / "reports" / "domain_spec.json")
        self.save_markdown_report(
            spec,
            summary,
            self._project_root / "reports" / "domain_reformulation.md",
        )
        self.update_config_domain(spec)
        self.save_context_memory(spec, summary)
        return spec

    def _get_client(self) -> Any | None:
        """Lazily initialize the Gemini SDK client."""
        if self._client is not None:
            return self._client
        if not self.is_available():
            return None

        try:
            genai = importlib.import_module("google.genai")
            self._client = genai.Client(api_key=self._api_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to initialize Gemini client lazily: {}", exc)
            return None

    def _generate_json_response(self, prompt: str) -> dict[str, Any]:
        """Call Gemini and parse a JSON-only response."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Gemini client is not available")

        types = importlib.import_module("google.genai.types")
        response = client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            ),
        )
        raw_text = (getattr(response, "text", "") or "").strip()
        if not raw_text:
            raise ValueError("Gemini returned an empty response")

        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
        if fenced_match:
            raw_text = fenced_match.group(1).strip()

        return json.loads(raw_text)

    def _build_prompt(
        self,
        topic: str,
        dataset_summary: dict[str, Any],
        current_classes: list[str],
    ) -> str:
        """Build a bounded prompt that contains only compact summary data."""
        payload = deepcopy(dataset_summary)
        memory_summary = ContextMemory(
            self._project_root / "reports" / "context_memory.json"
        ).get_summary_for_llm(max_chars=300)
        if memory_summary:
            payload["context_memory_summary"] = memory_summary

        summary_json = self._compact_summary_json(payload)
        prompt = (
            "You are designing a domain specification for an educational ML pipeline.\n"
            "Project focus must remain in sailing, yacht navigation, seamanship, safety,"
            " weather, licensing, and onboard equipment.\n"
            "Important constraints:\n"
            "- The current dataset contains noisy and partially off-topic rows.\n"
            "- Do not turn the task into emotion classification.\n"
            "- Recommend 5 to 7 classes suitable for future zero-shot annotation.\n"
            "- Always include review_label exactly as 'other_or_offtopic'.\n"
            "- Evaluate every source in source_fit with fit values high, medium, or low.\n"
            "- Use only the summary information provided below.\n"
            "- Return JSON only with keys: normalized_topic, recommended_classes,"
            " keywords_by_class, collection_queries, source_fit, review_label,"
            " annotation_guidelines, risks, llm_notes.\n"
            f"Current topic: {topic}\n"
            f"Current classes: {json.dumps(current_classes, ensure_ascii=False)}\n"
            f"Dataset summary: {summary_json}\n"
        )
        return prompt[: self._max_prompt_chars]

    def _compact_summary_json(self, dataset_summary: dict[str, Any]) -> str:
        """Shrink the summary JSON until it fits the configured prompt budget."""
        payload = deepcopy(dataset_summary)
        overhead = 1100

        while True:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) + overhead <= self._max_prompt_chars:
                return encoded

            top_keywords_by_source = payload.get("top_keywords_by_source", {})
            if any(len(keywords) > 4 for keywords in top_keywords_by_source.values()):
                payload["top_keywords_by_source"] = {
                    source_name: keywords[:4]
                    for source_name, keywords in top_keywords_by_source.items()
                }
                continue

            top_keywords_global = payload.get("top_keywords_global", [])
            if len(top_keywords_global) > 8:
                payload["top_keywords_global"] = top_keywords_global[:8]
                continue

            source_distribution = payload.get("source_distribution", {})
            if len(source_distribution) > 5:
                payload["source_distribution"] = dict(list(source_distribution.items())[:5])
                continue

            payload.pop("context_memory_summary", None)
            minimal_payload = {
                "total_rows": payload.get("total_rows", 0),
                "columns": payload.get("columns", [])[:5],
                "source_distribution": dict(
                    list(payload.get("source_distribution", {}).items())[:3]
                ),
                "avg_text_len": payload.get("avg_text_len", 0),
                "pct_short_texts": payload.get("pct_short_texts", 0),
                "pct_long_texts": payload.get("pct_long_texts", 0),
                "top_keywords_global": payload.get("top_keywords_global", [])[:5],
                "current_topic": payload.get("current_topic", ""),
                "current_classes": payload.get("current_classes", [])[:7],
            }
            return json.dumps(
                minimal_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def _validate_spec(self, spec: Any) -> dict[str, Any] | None:
        """Validate and normalize the generated domain specification."""
        if not isinstance(spec, dict):
            return None

        normalized_topic = str(spec.get("normalized_topic", "")).strip()
        recommended_classes = self._safe_classes(spec.get("recommended_classes", []))
        review_label = str(spec.get("review_label", "")).strip()

        if not normalized_topic or not (5 <= len(recommended_classes) <= 7):
            return None
        if review_label != "other_or_offtopic":
            return None

        keywords_by_class = {
            class_name: [
                str(keyword).strip()
                for keyword in spec.get("keywords_by_class", {}).get(class_name, [])
                if str(keyword).strip()
            ][:8]
            for class_name in recommended_classes
        }
        collection_queries = {
            class_name: [
                str(query).strip()
                for query in spec.get("collection_queries", {}).get(class_name, [])
                if str(query).strip()
            ][:4]
            for class_name in recommended_classes
        }
        source_fit_raw = spec.get("source_fit", {})
        source_fit = {}
        if not isinstance(source_fit_raw, dict) or not source_fit_raw:
            return None

        for source_name, fit_info in source_fit_raw.items():
            if not isinstance(fit_info, dict):
                return None
            fit_value = str(fit_info.get("fit", "")).strip().lower()
            if fit_value not in {"high", "medium", "low"}:
                return None
            source_fit[str(source_name)] = {
                "fit": fit_value,
                "notes": str(fit_info.get("notes", "")).strip(),
            }

        annotation_guidelines = [
            str(item).strip()
            for item in spec.get("annotation_guidelines", [])
            if str(item).strip()
        ]
        risks = [
            str(item).strip() for item in spec.get("risks", []) if str(item).strip()
        ]

        if not annotation_guidelines or not risks:
            return None

        return {
            "normalized_topic": normalized_topic,
            "recommended_classes": recommended_classes,
            "keywords_by_class": keywords_by_class,
            "collection_queries": collection_queries,
            "source_fit": source_fit,
            "review_label": review_label,
            "annotation_guidelines": annotation_guidelines,
            "risks": risks,
            "llm_notes": str(spec.get("llm_notes", "")).strip(),
        }

    def _keywords_for_class(
        self,
        class_name: str,
        dataset_summary: dict[str, Any],
    ) -> list[str]:
        """Choose deterministic fallback keywords for a class."""
        class_key = class_name.strip().lower()
        defaults = self._DEFAULT_CLASS_KEYWORDS.get(
            class_key,
            dataset_summary.get("top_keywords_global", [])[:5],
        )
        if defaults:
            return defaults[:5]
        return ["sailing", "navigation", "safety"]

    def _safe_classes(self, values: Any) -> list[str]:
        """Normalize a class list to unique lowercase strings."""
        if not isinstance(values, list):
            return list(self._DEFAULT_CLASSES)
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            class_name = re.sub(r"\s+", "_", str(value).strip().lower())
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            normalized.append(class_name)
        return normalized or list(self._DEFAULT_CLASSES)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text locally with regex and stopword filtering."""
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        return [token for token in tokens if token not in self._STOPWORDS]
