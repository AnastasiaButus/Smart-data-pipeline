"""Gemini-based domain reformulation utilities for step 1.3."""

from __future__ import annotations

import importlib
import json
import os
import re
import time
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
        self.model = str(
            self._llm_cfg.get("model", "models/gemini-2.0-flash")
        )
        self._temperature = float(self._llm_cfg.get("temperature", 0.2))
        self._max_tokens = int(self._llm_cfg.get("max_tokens", 800))
        self._max_prompt_chars = int(self._llm_cfg.get("max_prompt_chars", 2000))
        self._fallback_models = self._build_model_chain(
            self.model,
            self._llm_cfg.get(
                "fallback_models",
                [
                    "models/gemini-2.5-flash",
                    "models/gemini-flash-latest",
                    "models/gemma-3-4b-it",
                ],
            ),
        )

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
            return self._compose_domain_spec(validated, topic, dataset_summary)
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

        keywords_by_class = {
            class_name: self._keywords_for_class(class_name, dataset_summary)
            for class_name in classes
        }
        collection_queries = self._build_collection_queries(topic, keywords_by_class)
        source_fit = self._build_source_fit(dataset_summary)

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

    def generate(self, prompt: str) -> str:
        """Generate raw text with retry logic and model fallback chain."""
        client = self._get_client()
        if client is None:
            return "{}"

        prompt = prompt[: min(self._max_prompt_chars, 800)]
        types = importlib.import_module("google.genai.types")
        last_error = ""

        for attempt in range(3):
            retriable_seen = False
            for model_name in self._fallback_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self._temperature,
                            max_output_tokens=self._max_tokens,
                            response_mime_type="application/json",
                        ),
                    )
                    text = (getattr(response, "text", "") or "").strip()
                    if text:
                        logger.info(
                            "Gemini response received from model {} on attempt {}",
                            model_name,
                            attempt + 1,
                        )
                        return text
                    last_error = f"Empty response from {model_name}"
                except Exception as exc:
                    last_error = str(exc)
                    if "503" in last_error or "429" in last_error:
                        retriable_seen = True
                        logger.warning(
                            "Model {} attempt {} hit retriable error: {}",
                            model_name,
                            attempt + 1,
                            exc,
                        )
                        continue
                    logger.error(
                        "Model {} failed with non-retriable error: {}",
                        model_name,
                        exc,
                    )
                    return "{}"

            if retriable_seen and attempt < 2:
                wait = 10 * (attempt + 1)
                logger.warning("Attempt {} failed, waiting {}s", attempt + 1, wait)
                time.sleep(wait)
            else:
                break

        logger.error("Gemini generate exhausted retries. Last error: {}", last_error)
        return "{}"

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Generate JSON and recover from common formatting noise."""
        raw_text = self.generate(prompt).strip()
        if not raw_text:
            return {}

        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw_text = raw_text[start : end + 1]

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Gemini JSON: {}. Raw head: {}",
                exc,
                raw_text[:200],
            )
            return {}

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
        return self.generate_json(prompt)

    def _build_prompt(
        self,
        topic: str,
        dataset_summary: dict[str, Any],
        current_classes: list[str],
    ) -> str:
        """Build a short JSON-only prompt capped at 800 characters."""
        payload = self._compact_summary_json(
            {
                "total_rows": dataset_summary.get("total_rows", 0),
                "source_distribution": dataset_summary.get("source_distribution", {}),
                "top_keywords_global": dataset_summary.get("top_keywords_global", []),
                "pct_short_texts": dataset_summary.get("pct_short_texts", 0),
                "pct_long_texts": dataset_summary.get("pct_long_texts", 0),
            }
        )

        prompt = (
            "JSON only. Sailing domain only, not emotions. "
            "Use keys: normalized_topic, recommended_classes, keywords_by_class, "
            "annotation_guidelines, risks, llm_notes. "
            "Use 5 lowercase classes. Use 3 short keywords per class. "
            "Use 2 short guidelines. Use 2 short risks. "
            "Keep llm_notes under 12 words. "
            f"Topic={topic}. Classes={json.dumps(current_classes[:7], separators=(',', ':'))}. "
            f"Data={payload}."
        )
        return prompt[:800]

    def _compact_summary_json(self, dataset_summary: dict[str, Any]) -> str:
        """Shrink summary JSON to a compact payload safe for prompt embedding."""
        payload = deepcopy(dataset_summary)
        source_distribution = payload.get("source_distribution", {})
        payload["source_distribution"] = dict(list(source_distribution.items())[:3])
        payload["top_keywords_global"] = payload.get("top_keywords_global", [])[:6]

        while True:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= 350:
                return encoded

            top_keywords_global = payload.get("top_keywords_global", [])
            if len(top_keywords_global) > 4:
                payload["top_keywords_global"] = top_keywords_global[:4]
                continue

            source_distribution = payload.get("source_distribution", {})
            if len(source_distribution) > 2:
                payload["source_distribution"] = dict(list(source_distribution.items())[:2])
                continue

            minimal_payload = {
                "total_rows": payload.get("total_rows", 0),
                "sources": payload.get("source_distribution", {}),
                "pct_short_texts": payload.get("pct_short_texts", 0),
                "pct_long_texts": payload.get("pct_long_texts", 0),
                "keywords": payload.get("top_keywords_global", [])[:4],
            }
            return json.dumps(
                minimal_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def _validate_spec(self, spec: Any) -> dict[str, Any] | None:
        """Validate and normalize the LLM-returned partial specification."""
        if not isinstance(spec, dict):
            return None

        normalized_topic = str(spec.get("normalized_topic", "")).strip()
        recommended_classes = self._safe_classes(spec.get("recommended_classes", []))

        if not normalized_topic or not (5 <= len(recommended_classes) <= 7):
            return None

        keywords_by_class = {
            class_name: [
                str(keyword).strip()
                for keyword in spec.get("keywords_by_class", {}).get(class_name, [])
                if str(keyword).strip()
            ][:8]
            for class_name in recommended_classes
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
            "annotation_guidelines": annotation_guidelines,
            "risks": risks,
            "llm_notes": str(spec.get("llm_notes", "")).strip(),
        }

    def _compose_domain_spec(
        self,
        partial_spec: dict[str, Any],
        topic: str,
        dataset_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose the final spec from LLM output plus local heuristics."""
        keywords_by_class = partial_spec.get("keywords_by_class", {})
        llm_notes = partial_spec.get("llm_notes", "").strip()
        if not llm_notes:
            llm_notes = "Gemini domain spec generated from compact dataset summary."

        return {
            "normalized_topic": partial_spec["normalized_topic"],
            "recommended_classes": partial_spec["recommended_classes"],
            "keywords_by_class": keywords_by_class,
            "collection_queries": self._build_collection_queries(topic, keywords_by_class),
            "source_fit": self._build_source_fit(dataset_summary),
            "review_label": "other_or_offtopic",
            "annotation_guidelines": partial_spec["annotation_guidelines"],
            "risks": partial_spec["risks"],
            "llm_notes": llm_notes,
        }

    def _build_source_fit(
        self,
        dataset_summary: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Build heuristic source fit metadata from the dataset summary."""
        source_fit: dict[str, dict[str, str]] = {}
        for source_name in dataset_summary.get("source_distribution", {}).keys():
            if source_name.startswith("huggingface_"):
                fit = "low"
                notes = (
                    "Useful for volume, but partially off-topic and prone to generic"
                    " emotion or sentiment language."
                )
            elif source_name.startswith("stackexchange_"):
                fit = "high"
                notes = "Operational sailing questions align well with downstream annotation."
            elif source_name.startswith("rss_") or source_name == "sailingforums":
                fit = "high"
                notes = "Domain-relevant editorial or forum content fits sailing workflows well."
            else:
                fit = "medium"
                notes = "Source relevance is uncertain and should be reviewed in HITL."
            source_fit[source_name] = {"fit": fit, "notes": notes}
        return source_fit

    def _build_collection_queries(
        self,
        topic: str,
        keywords_by_class: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Build collection queries locally from class names and keywords."""
        queries: dict[str, list[str]] = {}
        for class_name, keywords in keywords_by_class.items():
            lead_keyword = keywords[0] if keywords else class_name
            queries[class_name] = [
                f"{topic} {class_name}",
                f"sailing {lead_keyword}",
            ]
        return queries

    def _build_model_chain(self, primary_model: str, fallback_models: Any) -> list[str]:
        """Build a unique ordered model chain starting with the primary model."""
        ordered = [str(primary_model).strip()]
        if isinstance(fallback_models, list):
            ordered.extend(str(model).strip() for model in fallback_models if str(model).strip())

        unique: list[str] = []
        seen: set[str] = set()
        for model_name in ordered:
            if model_name and model_name not in seen:
                seen.add(model_name)
                unique.append(model_name)
        return unique

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
