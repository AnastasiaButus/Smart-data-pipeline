"""Report generation helpers for the Streamlit HITL dashboard."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HTML_STYLE = """
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px;
    background: #f8f9fa;
    color: #1a1a2e;
  }
  h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #0f3460;
  }
  h2 {
    font-size: 1.25rem;
    margin-top: 32px;
    color: #0f3460;
    border-bottom: 2px solid #0f3460;
    padding-bottom: 8px;
  }
  .card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .muted {
    color: #5f6b7a;
  }
  ul {
    line-height: 1.8;
  }
</style>
""".strip()


def generate_html_report(sections: dict[str, bool], data: dict[str, Any]) -> str:
    """Assemble a compact HTML report from selected sections."""
    parts = [
        "<html><head><meta charset='utf-8'><title>Smart Data Pipeline Report</title>",
        HTML_STYLE,
        "</head><body>",
        "<h1>Smart Data Pipeline Report</h1>",
        (
            "<p class='muted'>"
            f"Тема: {escape(str(data.get('topic', 'не задана')))} | "
            f"Шаги: {escape(', '.join(data.get('steps_completed', [])) or 'нет данных')}"
            "</p>"
        ),
    ]

    if sections.get("domain"):
        parts.append(
            "<div class='card'><h2>Описание задачи и домена</h2>"
            f"<p>Тема: <strong>{escape(str(data.get('topic', 'не задана')))}</strong></p>"
            f"<p>Классы: {escape(', '.join(data.get('classes', [])) or 'нет данных')}</p>"
            "</div>"
        )

    if sections.get("sources"):
        source_items = "".join(
            f"<li>{escape(str(name))}: {count}</li>"
            for name, count in data.get("source_distribution", {}).items()
        ) or "<li>Источники недоступны</li>"
        parts.append(
            "<div class='card'><h2>Источники данных</h2><ul>"
            f"{source_items}</ul></div>"
        )

    if sections.get("before_cleaning"):
        parts.append(
            "<div class='card'><h2>Данные до обработки</h2>"
            f"<p>Строк до чистки: <strong>{data.get('total_rows', 0)}</strong></p>"
            "</div>"
        )

    if sections.get("after_cleaning"):
        parts.append(
            "<div class='card'><h2>Данные после обработки</h2>"
            f"<p>Строк после чистки: <strong>{data.get('clean_rows', 0)}</strong></p>"
            f"<p>Строк после авторазметки: <strong>{data.get('annotated_rows', 0)}</strong></p>"
            "</div>"
        )

    if sections.get("compare"):
        parts.append(
            "<div class='card'><h2>Сравнение до/после</h2>"
            f"<p>Удалено строк: <strong>{data.get('rows_removed', 0)}</strong></p>"
            f"<p>Очередь на ручную проверку: <strong>{data.get('review_queue_rows', 0)}</strong></p>"
            "</div>"
        )

    if sections.get("llm_hypotheses"):
        hypotheses = "".join(
            f"<li>{escape(str(item))}</li>"
            for item in data.get("llm_hypotheses", [])
        ) or "<li>Гипотезы недоступны</li>"
        parts.append(
            "<div class='card'><h2>LLM-гипотезы</h2><ul>"
            f"{hypotheses}</ul></div>"
        )

    if sections.get("hitl_stats"):
        parts.append(
            "<div class='card'><h2>HITL статистика</h2>"
            f"<p>Проверено вручную: <strong>{data.get('reviewed_rows', 0)}</strong></p>"
            f"<p>Осталось в review queue: <strong>{data.get('review_queue_rows', 0) - data.get('reviewed_rows', 0)}</strong></p>"
            "</div>"
        )

    if sections.get("model_metrics"):
        parts.append(
            "<div class='card'><h2>Метрики модели</h2>"
            f"<p>Средняя уверенность авторазметки: <strong>{data.get('confidence_mean', 0.0)}</strong></p>"
            "</div>"
        )

    if sections.get("retrospective"):
        parts.append(
            "<div class='card'><h2>Ретроспектива</h2>"
            "<p>Основной риск текущего пайплайна — высокий объём review queue и доменный шум в HuggingFace-источниках.</p>"
            "</div>"
        )

    parts.append("</body></html>")
    return "\n".join(parts)


def generate_markdown_report(sections: dict[str, bool], data: dict[str, Any]) -> str:
    """Assemble a Markdown report from selected sections."""
    lines = [
        "# Smart Data Pipeline Report",
        "",
        f"Тема: **{data.get('topic', 'не задана')}**",
        f"Шаги: {', '.join(data.get('steps_completed', [])) or 'нет данных'}",
        "",
    ]

    if sections.get("domain"):
        lines.extend(
            [
                "## Описание задачи и домена",
                f"- Тема: {data.get('topic', 'не задана')}",
                f"- Классы: {', '.join(data.get('classes', [])) or 'нет данных'}",
                "",
            ]
        )

    if sections.get("sources"):
        lines.append("## Источники данных")
        for name, count in data.get("source_distribution", {}).items():
            lines.append(f"- {name}: {count}")
        lines.append("")

    if sections.get("before_cleaning"):
        lines.extend(
            [
                "## Данные до обработки",
                f"- Строк до чистки: {data.get('total_rows', 0)}",
                "",
            ]
        )

    if sections.get("after_cleaning"):
        lines.extend(
            [
                "## Данные после обработки",
                f"- Строк после чистки: {data.get('clean_rows', 0)}",
                f"- Строк после авторазметки: {data.get('annotated_rows', 0)}",
                "",
            ]
        )

    if sections.get("compare"):
        lines.extend(
            [
                "## Сравнение до/после",
                f"- Удалено строк: {data.get('rows_removed', 0)}",
                f"- Очередь на проверку: {data.get('review_queue_rows', 0)}",
                "",
            ]
        )

    if sections.get("llm_hypotheses"):
        lines.append("## LLM-гипотезы")
        for item in data.get("llm_hypotheses", []):
            lines.append(f"- {item}")
        lines.append("")

    if sections.get("hitl_stats"):
        lines.extend(
            [
                "## HITL статистика",
                f"- Проверено: {data.get('reviewed_rows', 0)}",
                f"- Осталось: {data.get('review_queue_rows', 0) - data.get('reviewed_rows', 0)}",
                "",
            ]
        )

    if sections.get("model_metrics"):
        lines.extend(
            [
                "## Метрики модели",
                f"- Средняя уверенность: {data.get('confidence_mean', 0.0)}",
                "",
            ]
        )

    if sections.get("retrospective"):
        lines.extend(
            [
                "## Ретроспектива",
                "- Главный риск: слишком большая очередь на ручную проверку и доменный шум.",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def send_telegram_report(summary: str, bot_token: str, chat_id: str) -> bool:
    """Send a short report to Telegram and fail gracefully on any error."""
    token = str(bot_token).strip()
    target_chat = str(chat_id).strip()
    if not token or not target_chat:
        logger.error("Telegram send skipped: token or chat_id is missing")
        return False

    text = str(summary).strip()[:4096]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": target_chat, "text": text},
            timeout=20,
        )
        if response.status_code != 200:
            logger.error(
                "Telegram send failed: status={}, body={}",
                response.status_code,
                response.text[:300],
            )
            return False
        return True
    except Exception as exc:
        logger.error("Telegram send failed: {}", exc)
        return False


def collect_report_data() -> dict[str, Any]:
    """Collect compact report inputs from available project artifacts."""
    config_path = PROJECT_ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    topic = str(config.get("domain", {}).get("topic", "не задана"))
    classes = list(config.get("domain", {}).get("classes", []))
    review_label = str(config.get("domain", {}).get("review_label", "other_or_offtopic"))

    raw_df = _read_parquet(PROJECT_ROOT / "data" / "raw" / "dataset.parquet")
    clean_df = _read_parquet(PROJECT_ROOT / "data" / "raw" / "dataset_clean.parquet")
    annotated_df = _read_parquet(PROJECT_ROOT / "data" / "labeled" / "annotated.parquet")
    review_df = _read_csv(PROJECT_ROOT / "data" / "review_queue.csv")
    context_memory = _read_json(PROJECT_ROOT / "reports" / "context_memory.json")
    domain_spec = _read_json(PROJECT_ROOT / "reports" / "domain_spec.json")
    quality_report = _read_text(PROJECT_ROOT / "reports" / "quality_report.md")
    hypotheses = _read_json(PROJECT_ROOT / "reports" / "eda_hypotheses.json")

    steps_completed = _extract_steps(context_memory)
    source_distribution = (
        raw_df["source"].value_counts().to_dict()
        if not raw_df.empty and "source" in raw_df.columns
        else {}
    )
    label_distribution = (
        annotated_df["label"].value_counts().to_dict()
        if not annotated_df.empty and "label" in annotated_df.columns
        else {}
    )
    reviewed_rows = (
        int(review_df["corrected_label"].fillna("").astype(str).str.strip().ne("").sum())
        if not review_df.empty and "corrected_label" in review_df.columns
        else 0
    )

    return {
        "topic": topic,
        "classes": classes,
        "review_label": review_label,
        "total_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "annotated_rows": int(len(annotated_df)),
        "review_queue_rows": int(len(review_df)),
        "reviewed_rows": reviewed_rows,
        "confidence_mean": round(
            float(annotated_df["confidence"].mean()),
            3,
        )
        if not annotated_df.empty and "confidence" in annotated_df.columns
        else 0.0,
        "source_distribution": source_distribution,
        "label_distribution": label_distribution,
        "steps_completed": steps_completed,
        "rows_removed": max(int(len(raw_df)) - int(len(clean_df)), 0),
        "quality_report_excerpt": quality_report[:500],
        "llm_hypotheses": hypotheses if isinstance(hypotheses, list) else [],
        "domain_spec": domain_spec if isinstance(domain_spec, dict) else {},
    }


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet safely and return an empty dataframe on failure."""
    if not path.exists():
        logger.warning("Parquet artifact not found: {}", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.error("Failed to read parquet {}: {}", path, exc)
        return pd.DataFrame()


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV safely and return an empty dataframe on failure."""
    if not path.exists():
        logger.warning("CSV artifact not found: {}", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read CSV {}: {}", path, exc)
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    """Read JSON safely and return an empty payload on failure."""
    if not path.exists():
        logger.warning("JSON artifact not found: {}", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read JSON {}: {}", path, exc)
        return {}


def _read_text(path: Path) -> str:
    """Read text safely and return an empty string on failure."""
    if not path.exists():
        logger.warning("Text artifact not found: {}", path)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read text {}: {}", path, exc)
        return ""


def _extract_steps(context_memory: dict[str, Any] | list[Any]) -> list[str]:
    """Extract completed steps from context memory payload."""
    if not isinstance(context_memory, dict):
        return []

    history = context_memory.get("history", {})
    if isinstance(history, dict) and history:
        return sorted(history.keys(), key=_step_sort_key)

    step = context_memory.get("step")
    return [str(step)] if step else []


def _step_sort_key(value: str) -> tuple[int, ...]:
    """Sort step ids like 1.4, 2.1, 3.1 predictably."""
    parts = []
    for chunk in str(value).split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)
