"""Streamlit HITL dashboard for smart-data-pipeline step 3.2."""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.context_memory import ContextMemory
from core.llm_client import GeminiLLMClient
from ui.report_generator import (
    collect_report_data,
    generate_html_report,
    generate_markdown_report,
    send_telegram_report,
)

st.set_page_config(
    page_title="Smart Data Pipeline",
    page_icon="⛵",
    layout="wide",
)

CONFIG_PATH = ROOT / "config.yaml"
RAW_DATASET_PATH = ROOT / "data" / "raw" / "dataset.parquet"
CLEAN_DATASET_PATH = ROOT / "data" / "raw" / "dataset_clean.parquet"
ANNOTATED_DATASET_PATH = ROOT / "data" / "labeled" / "annotated.parquet"
REVIEW_QUEUE_PATH = ROOT / "data" / "review_queue.csv"
EDA_REPORT_PATH = ROOT / "reports" / "eda_report.html"
EDA_METADATA_PATH = ROOT / "reports" / "eda_metadata.json"

SAILING_TOPIC = "sailing and yacht navigation"
SAILING_DEFAULT_CLASSES = [
    "navigation",
    "safety",
    "equipment",
    "weather",
    "licensing",
]
REVIEW_COLUMNS = [
    "id",
    "text",
    "label",
    "confidence",
    "source",
    "suggested_label",
    "corrected_label",
]


def load_config() -> dict[str, Any]:
    """Load config.yaml safely for UI state."""
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("Не удалось прочитать config.yaml: {}", exc)
        return {}


def write_config(config: dict[str, Any]) -> None:
    """Persist config.yaml preserving UTF-8 and key order."""
    CONFIG_PATH.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def persist_domain_settings(
    *, topic: str | None = None, classes: list[str] | None = None
) -> None:
    """Update domain settings in config.yaml."""
    cfg = load_config()
    domain_cfg = cfg.setdefault("domain", {})
    if topic is not None:
        domain_cfg["topic"] = topic
    if classes is not None:
        domain_cfg["classes"] = classes
    write_config(cfg)


def clear_topic_source_state() -> None:
    """Reset onboarding source state when the topic changes."""
    st.session_state["source_suggestions"] = []
    st.session_state["selected_sources"] = []
    st.session_state["selected_sources_draft"] = []
    st.session_state["selected_items"] = {}
    st.session_state["review_df"] = empty_review_df()
    st.session_state["messages"] = []
    st.session_state["generated_report_content"] = ""
    st.session_state.pop("confirmed_sources", None)


def run_eda_export_for_current_topic() -> tuple[bool, str]:
    """Rebuild the interactive EDA report for the current artifacts."""
    command = [sys.executable, str(ROOT / "notebooks" / "export_eda.py")]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined_output = "\n".join(
        part.strip() for part in [completed.stdout, completed.stderr] if str(part).strip()
    )
    return completed.returncode == 0, combined_output


def run_pipeline_for_current_topic() -> None:
    """Run the full pipeline for the currently selected topic and persist a user-facing status."""
    topic = str(st.session_state.get("topic", SAILING_TOPIC)).strip() or SAILING_TOPIC
    command = [sys.executable, str(ROOT / "pipeline" / "run_pipeline.py")]
    logger.info("Запуск pipeline из UI для темы '{}'", topic)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined_output = "\n".join(
        part.strip() for part in [completed.stdout, completed.stderr] if str(part).strip()
    )
    st.session_state["auto_run_pipeline_pending"] = False
    st.session_state["last_pipeline_run_output"] = combined_output
    st.session_state["last_pipeline_run_code"] = int(completed.returncode)

    if completed.returncode == 0:
        eda_ok, eda_output = run_eda_export_for_current_topic()
        detail_parts = [part for part in [combined_output] if part]
        if eda_output:
            detail_parts.append("=== EDA export ===\n" + eda_output)
        combined_details = "\n\n".join(detail_parts)
        notice = {
            "kind": "success",
            "message": (
                f"Pipeline для темы **{topic}** завершён. "
                "Данные, HITL и аналитика обновлены."
            ),
            "details": combined_details,
        }
        if "RESOURCE_EXHAUSTED" in combined_output:
            notice = {
                "kind": "warning",
                "message": (
                    f"Pipeline для темы **{topic}** завершён, но Gemini quota была исчерпана "
                    "и часть LLM-шагов отработала через fallback."
                ),
                "details": combined_details,
            }
        if "Training skipped:" in combined_output or "training was skipped" in combined_output:
            notice = {
                "kind": "warning",
                "message": (
                    f"Данные для темы **{topic}** обновлены, но финальное обучение пока пропущено. "
                    "Сначала проверьте примеры во вкладке HITL, затем запустите переобучение."
                ),
                "details": combined_details,
            }
        if not eda_ok:
            notice = {
                "kind": "warning",
                "message": (
                    f"Данные для темы **{topic}** обновлены, но EDA-отчёт не удалось пересобрать автоматически. "
                    "Во вкладке Аналитика можно повторить пересборку вручную."
                ),
                "details": combined_details,
            }
        st.session_state["pipeline_refresh_notice"] = notice
        ensure_review_state(force_reload=True)
        st.rerun()

    quota_exhausted = "RESOURCE_EXHAUSTED" in combined_output or "Quota exceeded" in combined_output
    if quota_exhausted:
        message = (
            f"Pipeline для темы **{topic}** не завершился: у Gemini закончилась квота. "
            "Подождите сброса лимита, подключите другой API key или платный план."
        )
    else:
        message = (
            f"Pipeline для темы **{topic}** завершился с ошибкой. "
            "Посмотрите лог ниже и попробуйте повторить запуск."
        )
    st.session_state["pipeline_refresh_notice"] = {
        "kind": "error",
        "message": message,
        "details": combined_output,
    }
    st.rerun()


def is_sailing_topic(topic: str) -> bool:
    """Return True when the topic belongs to the sailing/yachting domain."""
    topic_lower = str(topic or "").lower()
    return any(
        keyword in topic_lower
        for keyword in [
            "sail",
            "yacht",
            "boat",
            "ship",
            "marine",
            "nautical",
            "ocean",
            "sea",
            "naval",
        ]
    )


def fallback_classes_for_topic(topic: str) -> list[str]:
    """Return topic-aware fallback classes when LLM output is unavailable."""
    topic_lower = str(topic or "").lower()
    mappings = [
        (
            ["sail", "yacht", "boat", "ship", "marine", "nautical", "ocean", "sea", "naval"],
            ["navigation", "safety", "equipment", "weather", "licensing"],
        ),
        (["minecraft", "game", "gaming"], ["survival", "building", "redstone", "combat", "exploration"]),
        (
            ["medical", "health", "doctor", "disease", "hospital", "pharma"],
            ["diagnosis", "symptoms", "treatment", "medication", "prevention"],
        ),
        (
            ["legal", "law", "court", "judge", "lawyer"],
            ["contracts", "compliance", "litigation", "regulation", "case_law"],
        ),
        (
            ["food", "cook", "recipe", "restaurant", "cuisine", "chef"],
            ["ingredients", "recipes", "cooking_methods", "nutrition", "restaurants"],
        ),
        (
            ["finance", "stock", "crypto", "invest", "trading", "bank"],
            ["markets", "investing", "risk", "regulation", "analysis"],
        ),
        (
            ["tech", "software", "code", "program", "computer", "ai", "ml"],
            ["development", "architecture", "debugging", "deployment", "models"],
        ),
    ]
    for keywords, classes in mappings:
        if any(keyword in topic_lower for keyword in keywords):
            return classes

    tokens = [
        re.sub(r"[^a-z0-9]+", "", token.lower())
        for token in topic_lower.split()
        if re.sub(r"[^a-z0-9]+", "", token.lower())
    ]
    base = tokens[0] if tokens else "topic"
    return [
        f"{base}_basics",
        f"{base}_tools",
        f"{base}_workflows",
        f"{base}_issues",
        f"{base}_advanced",
    ]


def empty_review_df() -> pd.DataFrame:
    """Return an empty review queue dataframe with the expected schema."""
    return pd.DataFrame(columns=REVIEW_COLUMNS)


def normalize_topic_name(topic: str) -> str:
    """Normalize a topic string for consistent comparisons."""
    return re.sub(r"\s+", " ", str(topic or "").strip().lower())


def get_artifact_topic(config: dict[str, Any] | None = None) -> str:
    """Return the topic associated with the currently generated data artifacts."""
    cfg = config or load_config()
    domain_cfg = cfg.get("domain", {}) or {}
    artifact_topic = str(domain_cfg.get("normalized_topic", "")).strip()
    if artifact_topic:
        return artifact_topic
    return str(domain_cfg.get("topic", "")).strip()


def get_topic_data_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe whether current parquet/review artifacts match the selected topic."""
    cfg = config or load_config()
    current_topic = str(
        st.session_state.get(
            "topic",
            cfg.get("domain", {}).get("topic", SAILING_TOPIC),
        )
    ).strip()
    artifact_topic = get_artifact_topic(cfg)
    artifact_paths = [
        RAW_DATASET_PATH,
        CLEAN_DATASET_PATH,
        ANNOTATED_DATASET_PATH,
        REVIEW_QUEUE_PATH,
    ]
    artifacts_exist = any(path.exists() for path in artifact_paths)
    current_normalized = normalize_topic_name(current_topic)
    artifact_normalized = normalize_topic_name(artifact_topic)
    is_fresh = (
        not artifacts_exist
        or not artifact_normalized
        or current_normalized == artifact_normalized
    )
    return {
        "current_topic": current_topic,
        "artifact_topic": artifact_topic,
        "artifacts_exist": artifacts_exist,
        "is_fresh": is_fresh,
    }


def read_parquet_safe(path: Path) -> pd.DataFrame:
    """Read parquet safely and return an empty dataframe on failure."""
    if not path.exists():
        logger.warning("Файл parquet не найден: {}", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.error("Не удалось прочитать parquet {}: {}", path, exc)
        return pd.DataFrame()


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV safely and return an empty dataframe on failure."""
    if not path.exists():
        logger.warning("Файл CSV не найден: {}", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.error("Не удалось прочитать CSV {}: {}", path, exc)
        return pd.DataFrame()


def read_json_safe(path: Path) -> dict[str, Any] | list[Any]:
    """Read JSON safely and return an empty payload on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Не удалось прочитать JSON {}: {}", path, exc)
        return {}


def get_latest_artifact_mtime(paths: list[Path]) -> float:
    """Return the latest modification time among existing files."""
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else 0.0


def get_eda_report_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe whether the saved EDA HTML report matches the current topic and data."""
    topic_status = get_topic_data_status(config)
    if not topic_status["is_fresh"]:
        return {
            "exists": EDA_REPORT_PATH.exists(),
            "is_fresh": False,
            "reason": "data_stale",
            "report_topic": "",
            "current_topic": topic_status["current_topic"],
        }

    metadata = read_json_safe(EDA_METADATA_PATH)
    if not EDA_REPORT_PATH.exists() or not isinstance(metadata, dict):
        return {
            "exists": EDA_REPORT_PATH.exists(),
            "is_fresh": False,
            "reason": "missing",
            "report_topic": "",
            "current_topic": topic_status["current_topic"],
        }

    current_topic = normalize_topic_name(topic_status["current_topic"])
    report_topic = normalize_topic_name(
        str(metadata.get("normalized_topic") or metadata.get("topic") or "")
    )
    latest_dataset_mtime = get_latest_artifact_mtime(
        [RAW_DATASET_PATH, CLEAN_DATASET_PATH, ANNOTATED_DATASET_PATH]
    )
    metadata_dataset_mtime = max(
        float(metadata.get("raw_dataset_mtime", 0.0) or 0.0),
        float(metadata.get("clean_dataset_mtime", 0.0) or 0.0),
        float(metadata.get("annotated_dataset_mtime", 0.0) or 0.0),
    )
    if not report_topic or current_topic != report_topic:
        reason = "topic_mismatch"
    elif metadata_dataset_mtime < latest_dataset_mtime:
        reason = "outdated"
    else:
        reason = "fresh"

    return {
        "exists": EDA_REPORT_PATH.exists(),
        "is_fresh": reason == "fresh",
        "reason": reason,
        "report_topic": str(metadata.get("topic") or ""),
        "current_topic": topic_status["current_topic"],
    }


def humanize_source_name(source_name: str) -> str:
    """Map internal source ids to readable names for analytics."""
    source = str(source_name or "")
    if source.startswith("huggingface_"):
        return f"HuggingFace / {source.removeprefix('huggingface_').replace('_', ' ')}"
    if source.startswith("stackexchange_"):
        return f"StackExchange / {source.removeprefix('stackexchange_').replace('_', ' ')}"
    if source.startswith("rss_"):
        return f"RSS / {source.removeprefix('rss_')}"
    if source.startswith("topic_bootstrap_"):
        return f"Topic bootstrap / {source.removeprefix('topic_bootstrap_').replace('_', ' ')}"
    if source.startswith("kaggle_"):
        return f"Kaggle / {source.removeprefix('kaggle_').replace('_', ' ')}"
    if "forum" in source.lower():
        return f"Forum / {source}"
    return source.replace("_", " ")


def source_license_and_status(source_name: str) -> tuple[str, str]:
    """Infer a compact license/status description from the source id."""
    source = str(source_name or "").lower()
    if source.startswith("huggingface_"):
        return ("dataset license", "✅")
    if source.startswith("stackexchange_"):
        return ("CC BY-SA 4.0", "✅")
    if source.startswith("rss_"):
        return ("editorial use", "⚠️")
    if source.startswith("topic_bootstrap_"):
        return ("generated bootstrap", "✅")
    if source.startswith("kaggle_"):
        return ("varies", "✅")
    if "forum" in source:
        return ("robots.txt checked", "⚠️")
    return ("unknown", "—")


def build_source_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a source summary table from the current dataset instead of static rows."""
    if df.empty or "source" not in df.columns:
        return pd.DataFrame()

    source_counts = (
        df["source"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .rename_axis("source")
        .reset_index(name="Строк")
    )
    records: list[dict[str, Any]] = []
    for _, row in source_counts.iterrows():
        source_name = str(row["source"])
        license_type, scraping_status = source_license_and_status(source_name)
        records.append(
            {
                "Источник": humanize_source_name(source_name),
                "Строк": int(row["Строк"]),
                "Тип лицензии": license_type,
                "Статус источника": scraping_status,
            }
        )
    return pd.DataFrame(records)


def init_state() -> None:
    """Initialize persistent Streamlit session state."""
    config = load_config()
    domain = config.get("domain", {}) or {}
    annotation = config.get("annotation", {}) or {}
    topic = str(domain.get("topic", "")).strip()
    classes = [
        str(item).strip()
        for item in domain.get("classes", [])
        if str(item).strip()
    ]
    topic_default_classes = fallback_classes_for_topic(topic) if topic else []
    if (
        topic
        and not is_sailing_topic(topic)
        and classes == SAILING_DEFAULT_CLASSES
        and topic_default_classes
    ):
        classes = topic_default_classes
        persist_domain_settings(classes=classes)

    st.session_state.setdefault("topic", topic)
    st.session_state.setdefault("current_topic", topic)
    st.session_state.setdefault("last_saved_topic", topic)
    st.session_state.setdefault("editing_topic", not bool(topic))
    st.session_state.setdefault("current_classes", classes)
    st.session_state.setdefault(
        "review_label",
        str(domain.get("review_label", "other_or_offtopic")).strip()
        or "other_or_offtopic",
    )
    st.session_state.setdefault(
        "confidence_threshold",
        float(annotation.get("confidence_threshold", 0.7)),
    )
    st.session_state.setdefault("selected_sources", [])
    st.session_state.setdefault("selected_sources_draft", [])
    st.session_state.setdefault("source_suggestions", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("generated_report_content", "")
    st.session_state.setdefault("generated_report_type", "html")
    st.session_state.setdefault("skip_active_learning", False)
    st.session_state.setdefault("skip_hitl", False)
    st.session_state.setdefault("auto_run_pipeline_pending", False)
    st.session_state.setdefault("pipeline_refresh_notice", None)
    st.session_state.setdefault("last_pipeline_run_output", "")
    st.session_state.setdefault("last_pipeline_run_code", 0)
    ensure_review_state()


def ensure_review_state(force_reload: bool = False) -> None:
    """Load review queue into session state when available."""
    status = get_topic_data_status()
    if not status["is_fresh"]:
        st.session_state["review_df"] = empty_review_df()
        return
    if force_reload or "review_df" not in st.session_state:
        review_df = read_csv_safe(REVIEW_QUEUE_PATH)
        if review_df.empty:
            st.session_state["review_df"] = empty_review_df()
        else:
            if "corrected_label" not in review_df.columns:
                review_df["corrected_label"] = ""
            st.session_state["review_df"] = review_df

@st.dialog("🎯 Настройка темы классификации")
def topic_dialog():
    st.markdown(
        "Введите тему для классификации текстов. "
        "Рекомендуем на **английском языке**."
    )
    st.caption(
        "💡 Например: sailing, medical diagnosis, "
        "legal documents, food recipes"
    )
    new_topic = st.text_input(
        "Тема",
        value=st.session_state.get(
            "topic", "sailing and yacht navigation"),
        key="dialog_topic_input"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Применить", type="primary",
                     use_container_width=True):
            if new_topic.strip():
                normalized_topic = new_topic.strip()
                previous_topic = str(st.session_state.get("topic", "")).strip()
                if normalized_topic != previous_topic:
                    clear_topic_source_state()
                    default_classes = fallback_classes_for_topic(normalized_topic)
                    st.session_state["current_classes"] = default_classes
                    st.session_state["source_suggestions"] = heuristic_source_suggestions(
                        normalized_topic
                    ).get("sources", [])
                    st.session_state.pop("classes_text_synced_from_classes", None)
                    persist_domain_settings(
                        topic=normalized_topic,
                        classes=default_classes,
                    )
                    st.session_state["auto_run_pipeline_pending"] = True
                    st.session_state["pipeline_refresh_notice"] = None
                    st.session_state["last_pipeline_run_output"] = ""
                    st.session_state["last_pipeline_run_code"] = 0
                else:
                    persist_domain_settings(topic=normalized_topic)
                st.session_state["topic"] = normalized_topic
                st.session_state["current_topic"] = normalized_topic
                st.session_state["last_saved_topic"] = normalized_topic
                st.session_state["editing_topic"] = False
                st.rerun()
    with col2:
        if st.button("Отмена", use_container_width=True):
            st.session_state["editing_topic"] = False
            st.rerun()


@st.cache_resource
def get_llm_client() -> GeminiLLMClient:
    """Create a GeminiLLMClient for dashboard interactions."""
    return GeminiLLMClient(config_path=str(CONFIG_PATH))


def get_best_dataset() -> pd.DataFrame:
    """Return the richest currently available dataset for UI analytics."""
    if not get_topic_data_status()["is_fresh"]:
        return pd.DataFrame()
    for path in [ANNOTATED_DATASET_PATH, CLEAN_DATASET_PATH, RAW_DATASET_PATH]:
        df = read_parquet_safe(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def get_pipeline_stats() -> dict[str, Any]:
    """Compute compact pipeline stats for sidebar and analytics."""
    status = get_topic_data_status()
    if not status["is_fresh"]:
        return {
            "raw_rows": 0,
            "clean_rows": 0,
            "annotated_rows": 0,
            "review_total": 0,
            "reviewed": 0,
            "progress_value": 0.0,
            "annotated_df": pd.DataFrame(),
            "data_stale": True,
            "artifact_topic": status["artifact_topic"],
        }

    raw_df = read_parquet_safe(RAW_DATASET_PATH)
    clean_df = read_parquet_safe(CLEAN_DATASET_PATH)
    annotated_df = read_parquet_safe(ANNOTATED_DATASET_PATH)
    review_df = st.session_state.get("review_df", pd.DataFrame()).copy()

    reviewed = (
        int(review_df["corrected_label"].fillna("").astype(str).str.strip().ne("").sum())
        if not review_df.empty and "corrected_label" in review_df.columns
        else 0
    )
    review_total = int(len(review_df))
    review_ratio = (reviewed / review_total) if review_total else 0.0
    progress_value = round((3 + review_ratio) / 6, 2)
    return {
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "annotated_rows": int(len(annotated_df)),
        "review_total": review_total,
        "reviewed": reviewed,
        "progress_value": progress_value,
        "annotated_df": annotated_df,
        "data_stale": False,
        "artifact_topic": status["artifact_topic"],
    }


def compute_review_impact(threshold: float) -> tuple[int, float]:
    """Estimate queue size at the current confidence threshold."""
    if not get_topic_data_status()["is_fresh"]:
        return 0, 0.0
    annotated_df = read_parquet_safe(ANNOTATED_DATASET_PATH)
    if annotated_df.empty or "confidence" not in annotated_df.columns:
        return 0, 0.0
    confidence = pd.to_numeric(annotated_df["confidence"], errors="coerce").fillna(0.0)
    count = int((confidence < threshold).sum())
    pct = round((count / len(annotated_df)) * 100, 2) if len(annotated_df) else 0.0
    return count, pct


def heuristic_source_suggestions(topic: str) -> dict[str, Any]:
    """Return fallback source suggestions in Russian for onboarding."""
    safe_topic = topic or "text classification"
    if is_sailing_topic(safe_topic):
        return {
            "sources": [
                {
                    "name": "StackExchange / форумы",
                    "type": "Q&A / forum",
                    "url": "https://api.stackexchange.com",
                    "license": "CC BY-SA 4.0",
                    "estimated_rows": 100,
                    "risk_level": "low",
                },
                {
                    "name": "RSS отраслевых медиа",
                    "type": "RSS",
                    "url": "https://www.yachtingworld.com/feed",
                    "license": "editorial use",
                    "estimated_rows": 40,
                    "risk_level": "medium",
                },
                {
                    "name": "HuggingFace dataset",
                    "type": "dataset",
                    "url": "https://huggingface.co/datasets",
                    "license": "depends on dataset",
                    "estimated_rows": 200,
                    "risk_level": "low",
                },
            ],
            "hf_datasets": [f"{safe_topic} classification dataset"],
            "suggested_classes": fallback_classes_for_topic(safe_topic),
        }

    encoded_topic = quote_plus(safe_topic)
    return {
        "sources": [
            {
                "name": f"HuggingFace datasets: {safe_topic}",
                "type": "dataset",
                "url": f"https://huggingface.co/datasets?search={encoded_topic}",
                "license": "depends on dataset",
                "estimated_rows": 200,
                "risk_level": "low",
            },
            {
                "name": f"StackExchange / forums: {safe_topic}",
                "type": "Q&A / forum",
                "url": f"https://stackexchange.com/search?q={encoded_topic}",
                "license": "CC BY-SA 4.0",
                "estimated_rows": 80,
                "risk_level": "low",
            },
            {
                "name": f"Reddit / communities: {safe_topic}",
                "type": "forum",
                "url": f"https://www.reddit.com/search/?q={encoded_topic}",
                "license": "depends on source",
                "estimated_rows": 120,
                "risk_level": "medium",
            },
            {
                "name": f"News / blogs: {safe_topic}",
                "type": "media",
                "url": f"https://www.google.com/search?q={encoded_topic}+blog",
                "license": "depends on source",
                "estimated_rows": 40,
                "risk_level": "medium",
            },
            {
                "name": f"Documentation / references: {safe_topic}",
                "type": "docs",
                "url": f"https://www.google.com/search?q={encoded_topic}+documentation",
                "license": "depends on source",
                "estimated_rows": 50,
                "risk_level": "medium",
            },
        ],
        "hf_datasets": [f"{safe_topic} classification dataset"],
        "suggested_classes": fallback_classes_for_topic(safe_topic),
    }


def find_sources_with_llm(topic: str, llm_client: GeminiLLMClient) -> dict[str, Any]:
    """Use Gemini to suggest candidate sources or return a heuristic fallback."""
    prompt = (
        f'For topic: "{topic}", suggest in Russian JSON: '
        '{"sources":[{"name":"...","type":"...","url":"...","license":"...",'
        '"estimated_rows":100,"risk_level":"low"}],"hf_datasets":["..."],'
        '"suggested_classes":["class1","class2","class3","class4","class5"]} '
        "JSON only, max 5 sources."
    )[:800]
    result = llm_client.generate_json(prompt)
    if isinstance(result, dict) and result.get("sources"):
        return result
    logger.warning("LLM onboarding fallback used for source suggestions")
    return heuristic_source_suggestions(topic)


def get_topic_emoji(topic: str) -> str:
    """Return an emoji matching the current topic, with LLM fallback."""
    topic_lower = str(topic or "").lower()
    emoji_map = [
        (["sail", "yacht", "boat", "ship", "marine", "nautical", "ocean", "sea", "naval"], "⛵"),
        (["game", "gaming", "minecraft"], "🎮"),
        (["medical", "health", "doctor", "disease", "hospital", "pharma"], "🏥"),
        (["food", "cook", "recipe", "restaurant", "cuisine", "chef"], "🍳"),
        (["tech", "software", "code", "program", "computer", "ai", "ml"], "💻"),
        (["sport", "football", "soccer", "tennis", "basketball", "athlete"], "⚽"),
        (["finance", "stock", "crypto", "invest", "trading", "bank"], "📈"),
        (["travel", "tourism", "hotel", "flight", "destination"], "✈️"),
        (["science", "research", "physics", "chemistry", "biology"], "🔬"),
        (["music", "song", "artist", "band", "concert", "album"], "🎵"),
        (["film", "movie", "cinema", "actor", "director"], "🎬"),
        (["law", "legal", "court", "judge", "lawyer"], "⚖️"),
        (["climate", "weather", "environment", "nature", "ecology"], "🌍"),
    ]
    for keywords, emoji in emoji_map:
        if any(keyword in topic_lower for keyword in keywords):
            return emoji

    try:
        from dotenv import load_dotenv

        load_dotenv()
        client = GeminiLLMClient(config_path=str(CONFIG_PATH))
        if client.is_available():
            result = client.generate(
                f"Reply with ONLY one emoji that best represents this topic: '{topic}'. One emoji, nothing else.",
                use_pipeline_context=False,
            )
            candidate = result.strip()
            if candidate and len(candidate) <= 4:
                return candidate
    except Exception:
        pass
    return "📊"


def update_classes_with_llm(llm_client: GeminiLLMClient) -> None:
    """Refresh recommended classes from the current topic, not the legacy dataset domain."""
    topic = str(st.session_state.get("topic", "")).strip()
    if not topic:
        st.sidebar.warning("Сначала задайте тему классификации.")
        return

    fallback_classes = fallback_classes_for_topic(topic)
    prompt = (
        "JSON only. Return keys: classes, review_label, notes. "
        f'For topic "{topic}" propose exactly 5 short lowercase English class labels '
        "for text classification. Use snake_case labels, 1-3 words each. "
        "Classes must match the new topic itself, not any previous domain or dataset. "
        "Avoid generic labels like misc, general, other. "
        'Set review_label to "other_or_offtopic".'
    )[:800]

    result = llm_client.generate_json(prompt)
    proposed_classes: list[str] = []
    for item in result.get("classes") or result.get("suggested_classes") or []:
        class_name = re.sub(r"\s+", "_", str(item).strip().lower())
        if class_name and class_name not in proposed_classes:
            proposed_classes.append(class_name)

    if len(proposed_classes) < 5:
        proposed_classes = fallback_classes
    else:
        proposed_classes = proposed_classes[:5]

    review_label = str(result.get("review_label", "other_or_offtopic")).strip() or "other_or_offtopic"
    st.session_state["current_classes"] = proposed_classes
    st.session_state["review_label"] = review_label
    st.session_state.pop("classes_text_synced_from_classes", None)
    persist_domain_settings(classes=proposed_classes)
    st.session_state["last_domain_spec"] = {
        "normalized_topic": topic,
        "recommended_classes": proposed_classes,
        "review_label": review_label,
        "llm_notes": result.get("notes", "Topic-first class refresh from dashboard."),
    }
    logger.info("Классы обновлены через LLM/topic-first flow: {}", proposed_classes)


def style_source_table(df: pd.DataFrame) -> Any:
    """Apply row colors by scraping risk level."""
    def _row_style(row: pd.Series) -> list[str]:
        risk = str(row.get("Риск скрапинга", "")).lower()
        if "low" in risk:
            color = "#e9f7ef"
        elif "medium" in risk:
            color = "#fff7d6"
        else:
            color = "#fdecec"
        return [f"background-color: {color}" for _ in row]

    return df.style.apply(_row_style, axis=1)


def style_source_table(df: pd.DataFrame) -> Any:
    """Apply row colors by scraping risk level."""

    def _row_style(row: pd.Series) -> list[str]:
        colors = {
            "low": "background-color: #1a3a1a",
            "medium": "background-color: #3a3a1a",
            "high": "background-color: #3a1a1a",
        }
        risk = str(row.get("Риск скрапинга", "")).lower()
        color = colors.get(risk, "")
        return [color for _ in row]

    return df.style.apply(_row_style, axis=1)


def normalize_source_suggestions(sources_data: list[Any]) -> pd.DataFrame:
    """Normalize heterogeneous source payloads from Gemini into one table schema."""
    normalized: list[dict[str, Any]] = []

    for source in sources_data:
        payload = source if isinstance(source, dict) else {}
        normalized.append(
            {
                "Источник": payload.get("name")
                or payload.get("source")
                or payload.get("title")
                or str(source),
                "Тип": payload.get("type")
                or payload.get("source_type")
                or "—",
                "Лицензия": payload.get("license")
                or payload.get("license_type")
                or "—",
                "Строк (ориентир)": payload.get("estimated_rows")
                or payload.get("rows")
                or "~100-500",
                "Риск скрапинга": payload.get("risk_level")
                or payload.get("risk")
                or "medium",
            }
        )

    if not normalized:
        normalized = [
            {
                "Источник": "StackExchange / форумы",
                "Тип": "API/форум",
                "Лицензия": "CC BY-SA 4.0",
                "Строк (ориентир)": "~100-200",
                "Риск скрапинга": "low",
            },
            {
                "Источник": "RSS отраслевых медиа",
                "Тип": "RSS",
                "Лицензия": "editorial use",
                "Строк (ориентир)": "~50-100",
                "Риск скрапинга": "medium",
            },
            {
                "Источник": "HuggingFace dataset",
                "Тип": "dataset",
                "Лицензия": "depends on dataset",
                "Строк (ориентир)": "~300-500",
                "Риск скрапинга": "low",
            },
        ]

    return pd.DataFrame(normalized)


def render_sidebar(llm_client: GeminiLLMClient) -> float:
    """Render sidebar controls and return the active threshold."""
    stats = get_pipeline_stats()
    cfg = load_config()
    st.sidebar.header("⚙️ Pipeline Settings")

    current_topic = st.session_state.get(
        "topic",
        load_config().get("domain", {}).get(
            "topic", "sailing and yacht navigation"))
    st.sidebar.markdown(f"**{current_topic}**")
    if st.sidebar.button(
            "✏️ Изменить тему",
            key="open_topic_dialog",
            use_container_width=True):
        st.session_state["editing_topic"] = True
        st.rerun()

    if st.sidebar.button("🔄 Обновить классы через LLM", use_container_width=True):
        with st.spinner("Gemini уточняет тему и классы..."):
            update_classes_with_llm(llm_client)
        st.sidebar.success("Классы обновлены")

    if st.session_state.get("current_classes"):
        st.sidebar.caption(
            "Классы: " + ", ".join(st.session_state.get("current_classes", []))
        )

    current_classes_state = st.session_state.get(
        "current_classes",
        cfg.get("domain", {}).get("classes", []),
    )
    desired_classes_text = "\n".join(current_classes_state)
    if (
        "classes_text" not in st.session_state
        or st.session_state.get("classes_text_synced_from_classes")
        != desired_classes_text
    ):
        st.session_state["classes_text"] = desired_classes_text
        st.session_state["classes_text_synced_from_classes"] = desired_classes_text

    st.sidebar.markdown("**Классы классификации:**")
    classes_input = st.sidebar.text_area(
        "Редактировать классы (каждый с новой строки):",
        height=150,
        help=(
            "Можно отредактировать предложенные LLM классы "
            "или написать свои с нуля"
        ),
        key="classes_text",
    )
    st.sidebar.caption(
        "💡 Классы лучше писать на английском — "
        "zero-shot классификация работает точнее "
        "когда метки на том же языке что и модель."
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 Применить классы", key="apply_sidebar_classes"):
            new_classes = [
                class_name.strip()
                for class_name in classes_input.split("\n")
                if class_name.strip()
            ]
            if len(new_classes) >= 2:
                persist_domain_settings(classes=new_classes)
                st.session_state["current_classes"] = new_classes
                st.session_state.pop("classes_text_synced_from_classes", None)
                st.sidebar.success(f"✅ Сохранено {len(new_classes)} классов")
                st.rerun()
            else:
                st.sidebar.error("Минимум 2 класса!")

    with col2:
        if st.button("↩️ Сбросить к дефолту", key="reset_sidebar_classes"):
            default_classes = fallback_classes_for_topic(
                st.session_state.get(
                    "topic",
                    cfg.get("domain", {}).get(
                        "topic", "sailing and yacht navigation"
                    ),
                )
            )
            persist_domain_settings(classes=default_classes)
            st.session_state["current_classes"] = default_classes
            st.session_state.pop("classes_text_synced_from_classes", None)
            st.rerun()

    current = [
        class_name.strip()
        for class_name in classes_input.split("\n")
        if class_name.strip()
    ]
    if current:
        st.sidebar.markdown(" ".join(f"`{class_name}`" for class_name in current))

    threshold = st.sidebar.slider(
        "Порог уверенности",
        min_value=0.3,
        max_value=0.9,
        value=float(st.session_state.get("confidence_threshold", 0.7)),
        step=0.05,
    )
    st.session_state["confidence_threshold"] = threshold

    queue_count, queue_pct = compute_review_impact(threshold)
    st.sidebar.caption(
        f"При пороге {threshold:.2f}: {queue_count} строк на проверку ({queue_pct:.1f}%)"
    )
    if queue_count > 200:
        st.sidebar.warning("⚠️ Большая очередь! Рекомендуем снизить порог до 0.5")

    st.sidebar.subheader("Прогресс пайплайна")
    st.sidebar.progress(stats["progress_value"])
    st.sidebar.write(f"✅ Сбор данных ({stats['raw_rows']} строк)")
    st.sidebar.write(f"✅ Чистка данных ({stats['clean_rows']} строк)")
    st.sidebar.write(f"✅ Авторазметка ({stats['annotated_rows']} строк)")
    st.sidebar.write(f"⏳ HITL проверка ({stats['reviewed']}/{stats['review_total']} проверено)")
    st.sidebar.write("⬜ Active Learning")
    st.sidebar.write("⬜ Обучение модели")
    if stats.get("data_stale"):
        st.sidebar.warning(
            "Текущие данные относятся к теме "
            f"'{stats.get('artifact_topic') or 'неизвестно'}'. "
            "Для новой темы нужно заново прогнать pipeline."
        )

    st.session_state["skip_active_learning"] = st.sidebar.checkbox(
        "Пропустить Active Learning",
        value=st.session_state.get("skip_active_learning", False),
    )
    if st.session_state["skip_active_learning"]:
        st.sidebar.warning("Без Active Learning модель не получит цикл доразметки на неуверенных примерах.")

    st.session_state["skip_hitl"] = st.sidebar.checkbox(
        "Пропустить HITL проверку",
        value=st.session_state.get("skip_hitl", False),
    )
    if st.session_state["skip_hitl"]:
        st.sidebar.warning("Без HITL в обучение попадут шумные и спорные примеры из review_queue.")

    return threshold


def render_hitl_tab(all_labels: list[str]) -> None:
    """Render manual review workflow for the review queue."""
    st.subheader("🔍 Проверка меток (HITL ★)")
    status = get_topic_data_status()
    if not status["is_fresh"]:
        st.warning(
            "Очередь HITL для текущей темы ещё не создана. "
            f"На диске сейчас лежат артефакты для темы: "
            f"**{status['artifact_topic'] or 'неизвестно'}**."
        )
        st.info(
            "Сначала перезапустите pipeline/annotation для новой темы, "
            "после этого review queue заполнится заново."
        )
        st.code("python pipeline/run_pipeline.py")
        return
    ensure_review_state()
    review_df = st.session_state.get("review_df", pd.DataFrame()).copy()

    if review_df.empty:
        st.info("review_queue.csv не найден. Сначала запустите AnnotationAgent.")
        st.code(
            'python -c "from agents.annotation_agent import AnnotationAgent; '
            'AnnotationAgent().run()"'
        )
        return

    corrected = review_df["corrected_label"].fillna("").astype(str).str.strip()
    reviewed_count = int(corrected.ne("").sum())
    total = int(len(review_df))
    remaining = total - reviewed_count
    progress_pct = round((reviewed_count / total) * 100, 2) if total else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего на проверке", total)
    col2.metric("Проверено", reviewed_count)
    col3.metric("Осталось", remaining)
    col4.metric("Прогресс", f"{progress_pct:.1f}%")

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    class_options = ["Все"] + sorted(review_df["label"].dropna().astype(str).unique().tolist())
    source_options = ["Все"] + sorted(review_df["source"].dropna().astype(str).unique().tolist())
    selected_class = filter_col1.selectbox("Фильтр по классу", class_options)
    selected_source = filter_col2.selectbox("Фильтр по источнику", source_options)
    only_unreviewed = filter_col3.checkbox("Показать только непроверенные", value=True)

    filtered_df = review_df.copy()
    if selected_class != "Все":
        filtered_df = filtered_df.loc[filtered_df["label"].astype(str) == selected_class]
    if selected_source != "Все":
        filtered_df = filtered_df.loc[filtered_df["source"].astype(str) == selected_source]
    if only_unreviewed:
        filtered_df = filtered_df.loc[
            filtered_df["corrected_label"].fillna("").astype(str).str.strip().eq("")
        ]
    filtered_df = filtered_df.reset_index()

    page_size = 10
    total_pages = max((len(filtered_df) - 1) // page_size + 1, 1)
    page = st.number_input("Страница", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    stop = start + page_size
    page_df = filtered_df.iloc[start:stop]

    if page_df.empty:
        st.info("По выбранным фильтрам нет строк.")
    else:
        for _, row in page_df.iterrows():
            original_index = int(row["index"])
            source_name = str(row.get("source", "unknown"))
            confidence = float(row.get("confidence", 0.0))
            auto_label = str(row.get("label", st.session_state.get("review_label")))
            default_label = str(
                row.get("corrected_label", "") or auto_label or st.session_state.get("review_label")
            )

            with st.container(border=True):
                st.markdown(f"**[{source_name}]** `confidence: {confidence:.2f}`")
                st.write(str(row.get("text", "")))
                st.caption(f"Авто-метка: {auto_label}")
                corrected_choice = st.selectbox(
                    "Исправить метку",
                    all_labels,
                    index=all_labels.index(default_label)
                    if default_label in all_labels
                    else 0,
                    key=f"review_choice_{original_index}",
                )
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("✅ Принять", key=f"accept_{original_index}"):
                    st.session_state["review_df"].at[original_index, "corrected_label"] = auto_label
                    st.rerun()
                if btn_col2.button("✏️ Исправить", key=f"fix_{original_index}"):
                    st.session_state["review_df"].at[original_index, "corrected_label"] = corrected_choice
                    st.rerun()

    action_col1, action_col2 = st.columns([1, 1])
    if action_col1.button("💾 Сохранить проверенные", key="save_review_queue_button"):
        review_payload = st.session_state["review_df"].copy()
        REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        review_payload.to_csv(REVIEW_QUEUE_PATH, index=False, encoding="utf-8")
        saved_count = int(
            review_payload["corrected_label"].fillna("").astype(str).str.strip().ne("").sum()
        )
        logger.info("review_queue.csv сохранён, исправлений={}", saved_count)
        st.success(f"Сохранено {saved_count} исправлений")

    action_col2.download_button(
        "📥 Скачать review_queue.csv",
        data=st.session_state["review_df"].to_csv(index=False).encode("utf-8"),
        file_name="review_queue.csv",
        mime="text/csv",
    )
    st.divider()
    st.subheader("🚀 Переобучить модель с исправлениями")

    corrected_count = int(
        review_df["corrected_label"].fillna("").astype(str).str.strip().ne("").sum()
    )
    st.info(f"Готово к обучению: {corrected_count} исправлений")

    if st.button("🔄 Запустить переобучение", key="retrain_from_hitl_button"):
        if not ANNOTATED_DATASET_PATH.exists():
            st.warning("annotated.parquet не найден. Сначала выполните AnnotationAgent.")
        else:
            with st.spinner("Обучаю модель..."):
                from core.model_wrapper import ModelWrapper

                base_df = pd.read_parquet(ANNOTATED_DATASET_PATH)
                corrected = review_df.loc[
                    review_df["corrected_label"].fillna("").astype(str).str.strip().ne("")
                ][["id", "text", "corrected_label", "source"]].rename(
                    columns={"corrected_label": "label"}
                )
                corrected["confidence"] = 1.0
                corrected["label_source"] = "hitl"

                combined = pd.concat([base_df, corrected], ignore_index=True)
                wrapper = ModelWrapper()
                metrics = wrapper.fit(combined)

            st.success("✅ Модель переобучена!")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(
                "Accuracy",
                f"{metrics['accuracy']:.3f}",
                delta=f"{metrics['accuracy'] - 0.50:+.3f} vs baseline",
            )
            metric_col2.metric(
                "F1 macro",
                f"{metrics['f1_macro']:.3f}",
                delta=f"{metrics['f1_macro'] - 0.43:+.3f} vs baseline",
            )
            metric_col3.metric("N train", int(metrics.get("n_train", 0)))

            st.subheader("F1 по классам")
            fig = px.bar(
                x=list(metrics["f1_per_class"].keys()),
                y=list(metrics["f1_per_class"].values()),
                labels={"x": "Класс", "y": "F1"},
                color=list(metrics["f1_per_class"].values()),
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_analytics_tab(threshold: float) -> None:
    """Render annotation analytics and report builder."""
    st.subheader("📊 Аналитика")
    status = get_topic_data_status()
    if not status["is_fresh"]:
        st.warning(
            "Аналитические артефакты ещё относятся к теме "
            f"**{status['artifact_topic'] or 'неизвестно'}**. "
            "Для новой темы сначала нужен новый прогон pipeline."
        )
        st.code("python pipeline/run_pipeline.py")
        return
    annotated_df = read_parquet_safe(ANNOTATED_DATASET_PATH)

    if annotated_df.empty:
        st.info("annotated.parquet не найден. Сначала выполните AnnotationAgent.")
    else:
        annotated_df = annotated_df.copy()
        annotated_df["label"] = annotated_df["label"].fillna("unknown").astype(str)
        annotated_df["confidence"] = pd.to_numeric(
            annotated_df["confidence"],
            errors="coerce",
        ).fillna(0.0)

        label_df = (
            annotated_df["label"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
        )
        fig_labels = px.bar(
            label_df,
            x="label",
            y="count",
            color="label",
            title="Распределение меток",
        )
        st.plotly_chart(fig_labels, use_container_width=True)

        low_conf_count = int((annotated_df["confidence"] < threshold).sum())
        fig_conf = px.histogram(
            annotated_df,
            x="confidence",
            nbins=30,
            title="Распределение confidence",
        )
        fig_conf.add_vline(x=threshold, line_dash="dash", line_color="#d62728")
        fig_conf.add_annotation(
            x=threshold,
            y=0.95,
            yref="paper",
            text=f"{low_conf_count} строк ниже порога",
            showarrow=False,
            bgcolor="white",
            bordercolor="#d62728",
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    source_table_df = build_source_table(get_best_dataset())
    if not source_table_df.empty:
        st.markdown("**Источники в текущем датасете**")
        st.dataframe(source_table_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**📊 Расширенный интерактивный EDA отчёт**")
    st.caption(
        "Полный отчёт с 8 графиками (zoom, hover, pan), "
        "WordCloud и LLM-гипотезами. "
        "Скачайте и откройте в браузере — "
        "работает без интернета."
    )

    eda_status = get_eda_report_status()
    if not eda_status["is_fresh"]:
        if eda_status["reason"] == "topic_mismatch":
            st.warning(
                "Сохранённый EDA-отчёт относится к теме "
                f"**{eda_status['report_topic'] or 'неизвестно'}**, а сейчас выбрана "
                f"**{eda_status['current_topic']}**. Пересоберите отчёт ниже."
            )
        elif eda_status["reason"] == "outdated":
            st.warning(
                "EDA-отчёт устарел: данные уже обновились, а HTML ещё нет. "
                "Нажмите кнопку ниже, чтобы пересобрать отчёт на текущих артефактах."
            )
        else:
            st.info(
                "EDA-отчёт ещё не создан для текущей темы. "
                "Его можно собрать прямо из этой вкладки."
            )

        if st.button("♻️ Пересобрать EDA отчёт", key="rebuild_eda_report"):
            with st.spinner("Пересобираю интерактивный EDA-отчёт..."):
                ok, output = run_eda_export_for_current_topic()
            if ok:
                st.success("EDA-отчёт пересобран для текущей темы.")
                st.session_state["pipeline_refresh_notice"] = {
                    "kind": "success",
                    "message": "EDA-отчёт обновлён на свежих данных.",
                    "details": output,
                }
                st.rerun()
            st.error("Не удалось пересобрать EDA-отчёт. Лог ниже.")
            with st.expander("Показать лог пересборки EDA"):
                st.code(output or "Лог пуст")

    eda_path = pathlib.Path("reports/eda_report.html")
    if eda_path.exists() and get_eda_report_status()["is_fresh"]:
        with open(eda_path, "rb") as report_file:
            st.download_button(
                label="⬇️ Скачать EDA отчёт (HTML, интерактивный)",
                data=report_file.read(),
                file_name="eda_report.html",
                mime="text/html",
                help="После скачивания откройте файл в Chrome/Firefox для просмотра",
            )
    else:
        st.warning(
            "Свежий EDA-отчёт пока недоступен. "
            "Сначала пересоберите его кнопкой выше."
        )

    st.divider()
    st.subheader("📓 Jupyter ноутбуки")

    notebook_col1, notebook_col2 = st.columns(2)

    with notebook_col1:
        st.markdown("**EDA ноутбук**")
        st.caption("Исходный анализ данных с графиками")
        st.code(
            "$env:JUPYTER_CONFIG_DIR = \"$PWD\\.jupyter_runtime\"\n"
            ".venv\\Scripts\\python.exe -m jupyter notebook notebooks/eda.ipynb",
            language="powershell",
        )
        st.caption(
            "💡 Эта команда использует Python из проекта и обходит сломанный "
            "глобальный Jupyter config из Anaconda."
        )

    with notebook_col2:
        st.markdown("**AL эксперимент**")
        st.caption("Сравнение стратегий Active Learning")
        st.code(
            "$env:JUPYTER_CONFIG_DIR = \"$PWD\\.jupyter_runtime\"\n"
            ".venv\\Scripts\\python.exe -m jupyter notebook notebooks/al_experiment.ipynb",
            language="powershell",
        )
        st.caption(
            "💡 Для второго ноутбука используйте ту же схему: локальный config + "
            "Jupyter из `.venv`, а не системный `jupyter.exe`."
        )

    st.info(
        "💡 Запускайте команды из папки проекта. "
        "Они изолируют Jupyter от глобального пользовательского конфига и открывают ноутбук "
        "в корректном окружении проекта."
    )

    st.subheader("📋 Сформировать отчёт")
    section_col1, section_col2 = st.columns(2)
    with section_col1:
        section_domain = st.checkbox("Описание задачи и домена", value=True)
        section_sources = st.checkbox("Источники данных", value=True)
        section_before = st.checkbox("Данные до обработки", value=True)
        section_after = st.checkbox("Данные после обработки", value=True)
        section_compare = st.checkbox("Сравнение до/после", value=False)
    with section_col2:
        section_hypotheses = st.checkbox("LLM-гипотезы", value=True)
        section_hitl = st.checkbox("HITL статистика", value=False)
        section_model = st.checkbox("Метрики модели", value=True)
        section_retro = st.checkbox("Ретроспектива", value=False)

    export_format = st.radio(
        "Формат выгрузки",
        options=["HTML", "Markdown", "Telegram"],
        horizontal=True,
    )
    telegram_token_default = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_token = ""
    chat_id = ""
    if export_format == "Telegram":
        telegram_token = st.text_input(
            "Bot Token",
            value=telegram_token_default,
            type="password",
        )
        chat_id = st.text_input("Chat ID", value="")

    sections = {
        "domain": section_domain,
        "sources": section_sources,
        "before_cleaning": section_before,
        "after_cleaning": section_after,
        "compare": section_compare,
        "llm_hypotheses": section_hypotheses,
        "hitl_stats": section_hitl,
        "model_metrics": section_model,
        "retrospective": section_retro,
    }

    if st.button("📤 Сгенерировать и скачать", key="build_report_button"):
        report_data = collect_report_data()
        if export_format == "HTML":
            content = generate_html_report(sections, report_data)
            st.session_state["generated_report_content"] = content
            st.session_state["generated_report_type"] = "html"
            st.success("HTML-отчёт готов")
        elif export_format == "Markdown":
            content = generate_markdown_report(sections, report_data)
            st.session_state["generated_report_content"] = content
            st.session_state["generated_report_type"] = "md"
            st.success("Markdown-отчёт готов")
        else:
            summary = generate_markdown_report(sections, report_data)
            ok = send_telegram_report(summary, telegram_token, chat_id)
            if ok:
                st.success("Отчёт отправлен в Telegram")
            else:
                st.error("Не удалось отправить отчёт в Telegram")

    generated_content = st.session_state.get("generated_report_content", "")
    generated_type = st.session_state.get("generated_report_type", "html")
    if generated_content:
        mime = "text/html" if generated_type == "html" else "text/markdown"
        filename = (
            "smart_data_pipeline_report.html"
            if generated_type == "html"
            else "smart_data_pipeline_report.md"
        )
        st.download_button(
            "⬇️ Скачать отчёт",
            data=generated_content.encode("utf-8"),
            file_name=filename,
            mime=mime,
        )


def fallback_chat_answer(question: str, report_data: dict[str, Any]) -> str:
    """Build a short Russian fallback answer for chat interactions."""
    question_lower = question.lower()
    if "порог" in question_lower:
        return (
            f"Сейчас средняя уверенность около {report_data.get('confidence_mean', 0.0)}, "
            "а очередь на ручную проверку остаётся большой. Практически стоит тестировать порог 0.5-0.6, "
            "чтобы снизить объём review_queue без полной потери качества."
        )
    if "other_or_offtopic" in question_lower:
        return (
            "Класс other_or_offtopic доминирует, потому что в датасете много нетематических HuggingFace-строк "
            "и модель осторожно отправляет неоднозначные тексты в review."
        )
    if "гипотез" in question_lower:
        return (
            "Главная гипотеза: модель сначала отделяет доменный sailing-контент от общего шумного текста, "
            "а уже затем различает navigation, safety, equipment, weather и licensing."
        )
    return (
        "Сейчас я вижу высокий объём review queue и заметную долю other_or_offtopic. "
        "Следующий практический шаг — подобрать порог уверенности и проверить несколько десятков примеров вручную."
    )


def answer_with_llm(question: str, llm_client: GeminiLLMClient) -> str:
    """Answer dashboard chat questions with Gemini or a Russian fallback."""
    report_data = collect_report_data()
    context_summary = ContextMemory(
        path=str(ROOT / "reports" / "context_memory.json")
    ).get_summary_for_llm()
    compact_metrics = {
        "topic": st.session_state.get("topic", report_data.get("topic", "")),
        "rows": report_data.get("annotated_rows", 0) or report_data.get("total_rows", 0),
        "classes": st.session_state.get("current_classes", []),
        "review_queue": report_data.get("review_queue_rows", 0),
        "confidence_mean": report_data.get("confidence_mean", 0.0),
    }
    prompt = (
        "Ты эксперт по ML и анализу данных. "
        f"Контекст проекта: {context_summary}. "
        f"Метрики: {json.dumps(compact_metrics, ensure_ascii=False, separators=(',', ':'))}. "
        f"Вопрос пользователя: {question}. "
        "Отвечай на русском, кратко и по делу."
    )[:800]

    response = llm_client.generate(prompt).strip()
    if not response or response == "{}":
        logger.warning("LLM chat fallback used")
        return fallback_chat_answer(question, report_data)
    return response


def handle_chat_prompt(prompt: str, llm_client: GeminiLLMClient) -> None:
    """Append a user message, generate an answer, and rerun the app."""
    st.session_state["messages"].append({"role": "user", "content": prompt})
    answer = answer_with_llm(prompt, llm_client)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.rerun()


def render_chat_tab(llm_client: GeminiLLMClient) -> None:
    """Render the dashboard chat tab."""
    st.subheader("💬 Обсудить данные с Gemini")
    status = get_topic_data_status()
    if not status["is_fresh"]:
        st.warning(
            "Чат временно отключён для новой темы, потому что данные и отчёты "
            f"ещё относятся к теме **{status['artifact_topic'] or 'неизвестно'}**."
        )
        st.info("Сначала прогоните pipeline заново, затем чат будет отвечать уже по новым данным.")
        return
    report_data = collect_report_data()
    topic = st.session_state.get("topic", report_data.get("topic", "не задана"))
    row_count = report_data.get("annotated_rows", 0) or report_data.get("total_rows", 0)
    classes = st.session_state.get("current_classes", report_data.get("classes", []))
    st.info(f"Контекст: {topic}, {row_count} строк, классы: {', '.join(classes)}")

    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    if quick_col1.button("Объясни гипотезы"):
        handle_chat_prompt("Объясни гипотезы для текущего датасета.", llm_client)
    if quick_col2.button("Что улучшить?"):
        handle_chat_prompt("Что улучшить в текущем пайплайне и данных?", llm_client)
    if quick_col3.button("Почему так много other_or_offtopic?"):
        handle_chat_prompt("Почему так много other_or_offtopic?", llm_client)
    if quick_col4.button("Какой порог confidence выбрать?"):
        handle_chat_prompt("Какой порог confidence выбрать для review queue?", llm_client)

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Задайте вопрос о данных или гипотезах...")
    if user_input:
        handle_chat_prompt(user_input, llm_client)


PERMISSION_LABELS = {
    "low": "✅ Свободно",
    "medium": "⚠️ С оговорками",
    "high": "🚫 Ограничено",
    "✅ Свободно": "✅ Свободно",
    "⚠️ С оговорками": "⚠️ С оговорками",
    "🚫 Ограничено": "🚫 Ограничено",
}


def normalize_source_suggestions(sources_data: list[Any]) -> pd.DataFrame:
    """Normalize heterogeneous source payloads into one stable onboarding schema."""
    normalized: list[dict[str, Any]] = []
    for source in sources_data:
        payload = source if isinstance(source, dict) else {}
        item_name = (
            payload.get("name")
            or payload.get("source")
            or payload.get("title")
            or str(source)
        )
        item_type = payload.get("type") or payload.get("source_type") or "—"
        item_url = str(payload.get("url", "")).strip()
        if item_type == "dataset" and not item_url:
            if "/" in item_name:
                item_url = f"https://huggingface.co/datasets/{item_name}"
            else:
                item_url = (
                    "https://huggingface.co/search/full-text"
                    f"?q={str(item_name).replace(' ', '+')}"
                )
        normalized.append(
            {
                "Источник": item_name,
                "Тип": item_type,
                "Лицензия": payload.get("license")
                or payload.get("license_type")
                or "—",
                "Строк (ориентир)": payload.get("estimated_rows")
                or payload.get("rows")
                or "~100-500",
                "Уровень разрешения": PERMISSION_LABELS.get(
                    payload.get("risk_level") or payload.get("risk") or "medium",
                    "⚠️ С оговорками",
                ),
                "URL": item_url,
            }
        )

    if not normalized:
        normalized = [
            {
                "Источник": "StackExchange / форумы",
                "Тип": "API/форум",
                "Лицензия": "CC BY-SA 4.0",
                "Строк (ориентир)": "~100-200",
                "Уровень разрешения": "✅ Свободно",
                "URL": "https://api.stackexchange.com",
            },
            {
                "Источник": "RSS отраслевых медиа",
                "Тип": "RSS",
                "Лицензия": "editorial use",
                "Строк (ориентир)": "~50-100",
                "Уровень разрешения": "⚠️ С оговорками",
                "URL": "https://www.yachtingworld.com/feed",
            },
            {
                "Источник": "HuggingFace datasets",
                "Тип": "dataset",
                "Лицензия": "depends on dataset",
                "Строк (ориентир)": "~300-500",
                "Уровень разрешения": "✅ Свободно",
                "URL": "https://huggingface.co/datasets",
            },
        ]

    return pd.DataFrame(normalized)


def build_sources_detail(sources_data: list[Any], topic: str) -> dict[str, Any]:
    """Build onboarding source groups from the current topic suggestions."""
    effective_sources = sources_data or heuristic_source_suggestions(topic).get("sources", [])
    details: dict[str, Any] = {}

    for source in effective_sources:
        payload = source if isinstance(source, dict) else {}
        item_name = (
            payload.get("name")
            or payload.get("source")
            or payload.get("title")
            or str(source)
        )
        item_type = str(payload.get("type") or payload.get("source_type") or "other").lower()
        item_url = str(payload.get("url", "")).strip()
        item_rows = payload.get("estimated_rows") or payload.get("rows") or 100
        item_license = str(payload.get("license") or payload.get("license_type") or "mixed")
        item_risk = PERMISSION_LABELS.get(
            payload.get("risk_level") or payload.get("risk") or "medium",
            "⚠️ С оговорками",
        )

        if "dataset" in item_type:
            group_name = "Datasets / corpora"
            description = f"Датасеты и корпуса для темы: {topic}"
        elif any(tag in item_type for tag in ["forum", "community", "q&a", "api"]):
            group_name = "Communities / forums"
            description = f"Форумы, Q&A и комьюнити по теме: {topic}"
        elif any(tag in item_type for tag in ["rss", "media", "news", "blog", "docs"]):
            group_name = "Media / docs"
            description = f"Медиа, блоги и документация по теме: {topic}"
        else:
            group_name = "Additional sources"
            description = f"Дополнительные источники по теме: {topic}"

        group = details.setdefault(
            group_name,
            {
                "description": description,
                "license": item_license,
                "risk": item_risk,
                "items": [],
            },
        )
        if group["license"] != item_license:
            group["license"] = "mixed"
        if group["risk"] == "✅ Свободно" and item_risk != "✅ Свободно":
            group["risk"] = item_risk
        elif group["risk"] == "⚠️ С оговорками" and item_risk == "🚫 Ограничено":
            group["risk"] = item_risk

        group["items"].append(
            {
                "name": str(item_name),
                "url": item_url,
                "rows": item_rows,
                "enabled": item_risk != "🚫 Ограничено",
                "disabled": bool(payload.get("disabled", False)),
            }
        )

    if "Kaggle datasets" not in details:
        details["Kaggle datasets"] = {
            "description": "Поиск дополнительных текстовых датасетов на Kaggle",
            "license": "varies (CC / public domain)",
            "risk": "✅ Свободно",
            "items": [
                {
                    "name": f"Kaggle search: {topic}",
                    "url": f"https://www.kaggle.com/search?q={quote_plus(topic or 'text classification')}",
                    "rows": 200,
                    "enabled": False,
                    "disabled": True,
                }
            ],
        }

    return details


def render_onboarding_tab(llm_client: GeminiLLMClient) -> None:
    """Render onboarding flow for topic and source discovery."""
    current_topic = st.session_state.get(
        "topic",
        load_config().get("domain", {}).get("topic", "sailing and yacht navigation"),
    )
    status = get_topic_data_status()
    st.title(f"{get_topic_emoji(current_topic)} Smart Data Pipeline")

    notice = st.session_state.get("pipeline_refresh_notice")
    if isinstance(notice, dict) and notice.get("message"):
        notice_kind = str(notice.get("kind", "info")).lower()
        notice_message = str(notice.get("message", ""))
        if notice_kind == "success":
            st.success(notice_message)
        elif notice_kind == "warning":
            st.warning(notice_message)
        elif notice_kind == "error":
            st.error(notice_message)
        else:
            st.info(notice_message)
        if notice.get("details"):
            with st.expander("Показать лог запуска pipeline"):
                st.code(str(notice["details"]))

    if not status["is_fresh"]:
        with st.container(border=True):
            st.markdown("### 🔄 Нужно обновить данные под новую тему")
            st.markdown(
                f"Сейчас выбрана тема **{current_topic}**, "
                f"но на диске пока лежат артефакты для темы **{status['artifact_topic'] or 'неизвестно'}**."
            )
            st.caption(
                "Это нормально после смены темы: сначала нужно заново собрать тексты, "
                "разметить их и построить новую review queue."
            )

            step_col1, step_col2, step_col3 = st.columns(3)
            with step_col1:
                st.markdown("**1. Тема**")
                st.caption(f"Выбрана: {current_topic}")
            with step_col2:
                st.markdown("**2. Обновление данных**")
                st.caption("Нужно один раз прогнать pipeline")
            with step_col3:
                st.markdown("**3. Результат**")
                st.caption("После этого оживут HITL, аналитика и чат")

            action_col1, action_col2 = st.columns([1.4, 1])
            with action_col1:
                if st.button(
                    "▶ Обновить данные для этой темы",
                    type="primary",
                    key="run_pipeline_from_onboarding",
                    use_container_width=True,
                ):
                    with st.spinner("Запускаю pipeline. Это может занять несколько минут..."):
                        run_pipeline_for_current_topic()
                    return
            with action_col2:
                st.markdown("**Если хочешь вручную**")
                with st.expander("Показать команду"):
                    st.code("python pipeline/run_pipeline.py")

        if st.session_state.get("auto_run_pipeline_pending", False):
            st.info(
                "Тема изменена — автоматически обновляю данные. "
                "Подожди немного: после завершения здесь появится результат."
            )
            with st.spinner("Автоматически запускаю pipeline для новой темы..."):
                run_pipeline_for_current_topic()
            return

    st.subheader("Шаг 1. Проверьте тему и классы")
    with st.container(border=True):
        summary_col1, summary_col2 = st.columns([1.2, 1.8])
        with summary_col1:
            st.markdown(f"**🎯 Тема:** `{current_topic}`")
            data_status_text = "Свежие данные готовы" if status["is_fresh"] else "Нужно обновить данные"
            data_status_kind = "🟢" if status["is_fresh"] else "🟡"
            st.caption(f"{data_status_kind} {data_status_text}")
        with summary_col2:
            classes_preview = ", ".join(st.session_state.get("current_classes", []))
            st.markdown(f"**🏷️ Классы:** {classes_preview}")

        helper_col1, helper_col2 = st.columns([1, 2])
        with helper_col1:
            if st.button("Изменить тему", key="change_topic_button"):
                st.session_state["editing_topic"] = True
                st.rerun()
        with helper_col2:
            st.caption(
                "Если тема не подходит, нажми **✏️ Изменить тему** в боковой панели. "
                "Классы можно обновить кнопкой **🔄 Обновить классы через LLM** или поправить вручную."
            )

    if st.session_state.get("selected_sources"):
        st.success(
            "Выбраны источники: "
            + ", ".join(st.session_state.get("selected_sources", []))
        )

    if "selected_items" not in st.session_state:
        st.session_state["selected_items"] = {}

    st.subheader("Шаг 2. Подберите источники данных")
    st.caption(
        "Нажми кнопку ниже, чтобы подобрать источники под текущую тему, "
        "или используй уже предложенный набор и отметь нужные чекбоксами."
    )
    if st.button("🔍 Найти источники данных", key="find_sources_button"):
        with st.spinner("Gemini ищет источники..."):
            suggestion_payload = find_sources_with_llm(current_topic, llm_client)
        st.session_state["source_suggestions"] = suggestion_payload.get("sources", [])
        if suggestion_payload.get("suggested_classes"):
            refreshed_classes = [
                str(item).strip()
                for item in suggestion_payload.get("suggested_classes", [])
                if str(item).strip()
            ]
            if refreshed_classes:
                st.session_state["current_classes"] = refreshed_classes
                st.session_state.pop("classes_text_synced_from_classes", None)
                persist_domain_settings(classes=refreshed_classes)

    suggestions = st.session_state.get("source_suggestions", [])
    effective_suggestions = suggestions or heuristic_source_suggestions(current_topic).get(
        "sources", []
    )

    sources_detail = build_sources_detail(effective_suggestions, current_topic)
    if len(sources_detail) > 5:
        # TODO: пагинация при большом количестве дополнительных источников
        pass

    st.markdown("### 📦 Доступные источники данных")
    st.caption("Раскройте каждый источник чтобы выбрать конкретные датасеты и сайты")

    total_selected = 0
    total_rows = 0
    selected_labels: list[str] = []

    for source_name, source_data in sources_detail.items():
        risk_icon = {
            "✅ Свободно": "✅",
            "⚠️ С оговорками": "⚠️",
            "🚫 Ограничено": "🚫",
        }.get(source_data["risk"], "⚪")

        with st.expander(
            f"{risk_icon} **{source_name}** — {source_data['license']}"
        ):
            st.caption(source_data["description"])

            for item in source_data["items"]:
                item_key = f"{source_name}_{item['name']}"
                if item_key not in st.session_state["selected_items"]:
                    st.session_state["selected_items"][item_key] = bool(
                        item.get("enabled", True)
                    )

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    checked = st.checkbox(
                        item["name"],
                        value=st.session_state["selected_items"][item_key],
                        key=f"cb_{item_key}",
                        disabled=bool(item.get("disabled", False)),
                        help=(
                            "Датасет содержит числовые данные, не тексты. Недоступен для выбора."
                            if item.get("disabled", False)
                            else None
                        ),
                    )
                    st.session_state["selected_items"][item_key] = (
                        False if item.get("disabled", False) else checked
                    )
                with col2:
                    st.caption(f"~{item['rows']} строк")
                with col3:
                    item_url = str(item.get("url", "")).strip()
                    if item_url:
                        st.link_button("🔗", item_url, help="Открыть источник")
                    else:
                        st.caption("—")

                if st.session_state["selected_items"][item_key]:
                    total_selected += 1
                    try:
                        total_rows += int(item["rows"])
                    except Exception:
                        pass
                    selected_labels.append(f"{source_name} / {item['name']}")

    st.divider()
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Выбрано источников", total_selected)
    metric_col2.metric("Ожидаемых строк", f"~{total_rows}")

    if st.button("✅ Использовать выбранные источники", type="primary", key="confirm_source_selection"):
        selected = {
            key: value
            for key, value in st.session_state["selected_items"].items()
            if value
        }
        st.session_state["confirmed_sources"] = selected
        st.session_state["selected_sources"] = selected_labels
        st.session_state["editing_topic"] = False
        cfg_data = load_config()
        sources_cfg = cfg_data.setdefault("sources", {})
        sources_cfg["selected"] = selected_labels
        write_config(cfg_data)
        st.success(f"Сохранено {len(selected)} источников!")
        st.rerun()


def normalize_source_suggestions(sources_data: list[Any]) -> pd.DataFrame:
    """Normalize heterogeneous source payloads into one stable onboarding schema."""
    normalized: list[dict[str, Any]] = []
    for source in sources_data:
        payload = source if isinstance(source, dict) else {}
        item_name = (
            payload.get("name")
            or payload.get("source")
            or payload.get("title")
            or str(source)
        )
        item_type = payload.get("type") or payload.get("source_type") or "—"
        item_url = str(payload.get("url", "")).strip()
        if item_type == "dataset" and not item_url:
            if "/" in str(item_name):
                item_url = f"https://huggingface.co/datasets/{item_name}"
            else:
                item_url = (
                    "https://huggingface.co/search/full-text"
                    f"?q={str(item_name).replace(' ', '+')}"
                )

        normalized.append(
            {
                "Источник": item_name,
                "Тип": item_type,
                "Лицензия": payload.get("license")
                or payload.get("license_type")
                or "—",
                "Строк (ориентир)": payload.get("estimated_rows")
                or payload.get("rows")
                or "~100-500",
                "Уровень разрешения": PERMISSION_LABELS.get(
                    payload.get("risk_level") or payload.get("risk") or "medium",
                    "⚠️ С оговорками",
                ),
                "URL": item_url,
            }
        )

    if not normalized:
        normalized = [
            {
                "Источник": "StackExchange / форумы",
                "Тип": "API/форум",
                "Лицензия": "CC BY-SA 4.0",
                "Строк (ориентир)": "~100-200",
                "Уровень разрешения": "✅ Свободно",
                "URL": "https://api.stackexchange.com",
            },
            {
                "Источник": "RSS отраслевых медиа",
                "Тип": "RSS",
                "Лицензия": "editorial use",
                "Строк (ориентир)": "~50-100",
                "Уровень разрешения": "⚠️ С оговорками",
                "URL": "https://www.yachtingworld.com/feed",
            },
            {
                "Источник": "HuggingFace datasets",
                "Тип": "dataset",
                "Лицензия": "depends on dataset",
                "Строк (ориентир)": "~300-500",
                "Уровень разрешения": "✅ Свободно",
                "URL": "https://huggingface.co/datasets",
            },
        ]

    return pd.DataFrame(normalized)


def build_sources_detail(sources_data: list[Any]) -> dict[str, Any]:
    """Build detailed onboarding source groups with explicit dataset links."""
    details: dict[str, Any] = {
        "StackExchange / форумы": {
            "description": "Q&A форумы по теме",
            "license": "CC BY-SA 4.0",
            "risk": "✅ Свободно",
            "items": [
                {
                    "name": "sailing.stackexchange.com",
                    "url": "https://sailing.stackexchange.com",
                    "rows": 98,
                    "enabled": True,
                },
                {
                    "name": "outdoors.stackexchange.com",
                    "url": "https://outdoors.stackexchange.com",
                    "rows": 50,
                    "enabled": False,
                },
            ],
        },
        "HuggingFace datasets": {
            "description": "Открытые ML датасеты",
            "license": "зависит от датасета",
            "risk": "✅ Свободно",
            "items": [
                {
                    "name": "dair-ai/emotion",
                    "url": "https://huggingface.co/datasets/dair-ai/emotion",
                    "rows": 300,
                    "enabled": True,
                },
                {
                    "name": "mteb/tweet_sentiment_extraction",
                    "url": "https://huggingface.co/datasets/mteb/tweet_sentiment_extraction",
                    "rows": 300,
                    "enabled": True,
                },
            ],
        },
        "RSS отраслевых медиа": {
            "description": "Новости яхтинга и парусного спорта",
            "license": "editorial use",
            "risk": "⚠️ С оговорками",
            "items": [
                {
                    "name": "Yachting World",
                    "url": "https://www.yachtingworld.com/feed",
                    "rows": 30,
                    "enabled": True,
                },
                {
                    "name": "Cruising World",
                    "url": "https://www.cruisingworld.com/feed/",
                    "rows": 10,
                    "enabled": True,
                },
                {
                    "name": "Sail Magazine",
                    "url": "https://www.sailmagazine.com/feed",
                    "rows": 10,
                    "enabled": True,
                },
                {
                    "name": "48° North",
                    "url": "https://www.48north.com/feed/",
                    "rows": 10,
                    "enabled": True,
                },
            ],
        },
        "Форумы": {
            "description": "Тематические форумы яхтсменов",
            "license": "robots.txt checked",
            "risk": "⚠️ С оговорками",
            "items": [
                {
                    "name": "Sailing Forums",
                    "url": "https://www.sailingforums.com",
                    "rows": 20,
                    "enabled": True,
                }
            ],
        },
    }

    return details


def build_sources_detail(sources_data: list[Any], topic: str = "") -> dict[str, Any]:
    """Build onboarding source groups from current topic suggestions."""
    effective_sources = sources_data or heuristic_source_suggestions(topic).get("sources", [])
    details: dict[str, Any] = {}

    for source in effective_sources:
        payload = source if isinstance(source, dict) else {}
        item_name = (
            payload.get("name")
            or payload.get("source")
            or payload.get("title")
            or str(source)
        )
        item_type = str(payload.get("type") or payload.get("source_type") or "other").lower()
        item_url = str(payload.get("url", "")).strip()
        item_rows = payload.get("estimated_rows") or payload.get("rows") or 100
        item_license = str(payload.get("license") or payload.get("license_type") or "mixed")
        item_risk = PERMISSION_LABELS.get(
            payload.get("risk_level") or payload.get("risk") or "medium",
            "⚠️ С оговорками",
        )

        if "dataset" in item_type:
            group_name = "Datasets / corpora"
            description = f"Датасеты и корпуса для темы: {topic}"
        elif any(tag in item_type for tag in ["forum", "community", "q&a", "api"]):
            group_name = "Communities / forums"
            description = f"Форумы, Q&A и комьюнити по теме: {topic}"
        elif any(tag in item_type for tag in ["rss", "media", "news", "blog", "docs"]):
            group_name = "Media / docs"
            description = f"Медиа, блоги и документация по теме: {topic}"
        else:
            group_name = "Additional sources"
            description = f"Дополнительные источники по теме: {topic}"

        group = details.setdefault(
            group_name,
            {
                "description": description,
                "license": item_license,
                "risk": item_risk,
                "items": [],
            },
        )
        if group["license"] != item_license:
            group["license"] = "mixed"
        if group["risk"] == "✅ Свободно" and item_risk != "✅ Свободно":
            group["risk"] = item_risk
        elif group["risk"] == "⚠️ С оговорками" and item_risk == "🚫 Ограничено":
            group["risk"] = item_risk

        group["items"].append(
            {
                "name": str(item_name),
                "url": item_url,
                "rows": item_rows,
                "enabled": item_risk != "🚫 Ограничено",
                "disabled": bool(payload.get("disabled", False)),
            }
        )

    if "Kaggle datasets" not in details:
        details["Kaggle datasets"] = {
            "description": "Поиск дополнительных текстовых датасетов на Kaggle",
            "license": "varies (CC / public domain)",
            "risk": "✅ Свободно",
            "items": [
                {
                    "name": f"Kaggle search: {topic or 'text classification'}",
                    "url": f"https://www.kaggle.com/search?q={quote_plus(topic or 'text classification')}",
                    "rows": 200,
                    "enabled": False,
                    "disabled": True,
                }
            ],
        }

    return details


def main() -> None:
    """Run the Streamlit HITL dashboard."""
    init_state()
    if st.session_state.get("editing_topic", False):
        topic_dialog()
    llm_client = get_llm_client()
    threshold = render_sidebar(llm_client)

    all_labels = [
        *st.session_state.get("current_classes", []),
        st.session_state.get("review_label", "other_or_offtopic"),
    ]
    unique_labels: list[str] = []
    for label in all_labels:
        if label and label not in unique_labels:
            unique_labels.append(label)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚀 Онбординг", "🔍 Проверка меток (HITL ★)", "📊 Аналитика", "💬 Чат с LLM"]
    )
    with tab1:
        render_onboarding_tab(llm_client)
    with tab2:
        render_hitl_tab(unique_labels)
    with tab3:
        render_analytics_tab(threshold)
    with tab4:
        render_chat_tab(llm_client)


if __name__ == "__main__":
    main()
