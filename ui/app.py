"""Streamlit HITL dashboard for smart-data-pipeline step 3.2."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

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

LICENSE_ROWS = [
    {"Источник": "HuggingFace emotion", "Строк": 298, "Тип лицензии": "Apache 2.0", "Статус скрапинга": "✅"},
    {"Источник": "HuggingFace tweets", "Строк": 285, "Тип лицензии": "MIT", "Статус скрапинга": "✅"},
    {"Источник": "StackExchange", "Строк": 98, "Тип лицензии": "CC BY-SA 4.0", "Статус скрапинга": "✅"},
    {"Источник": "Yachting World RSS", "Строк": 30, "Тип лицензии": "editorial use", "Статус скрапинга": "⚠️"},
    {"Источник": "Sailing Forums", "Строк": 20, "Тип лицензии": "robots.txt checked", "Статус скрапинга": "⚠️"},
]


def load_config() -> dict[str, Any]:
    """Load config.yaml safely for UI state."""
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("Не удалось прочитать config.yaml: {}", exc)
        return {}


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

    st.session_state.setdefault("topic", topic)
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
    st.session_state.setdefault("source_suggestions", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("generated_report_content", "")
    st.session_state.setdefault("generated_report_type", "html")
    st.session_state.setdefault("skip_active_learning", False)
    st.session_state.setdefault("skip_hitl", False)
    ensure_review_state()


def ensure_review_state(force_reload: bool = False) -> None:
    """Load review queue into session state when available."""
    if force_reload or "review_df" not in st.session_state:
        review_df = read_csv_safe(REVIEW_QUEUE_PATH)
        if review_df.empty:
            st.session_state["review_df"] = pd.DataFrame(
                columns=[
                    "id",
                    "text",
                    "label",
                    "confidence",
                    "source",
                    "suggested_label",
                    "corrected_label",
                ]
            )
        else:
            if "corrected_label" not in review_df.columns:
                review_df["corrected_label"] = ""
            st.session_state["review_df"] = review_df


def build_llm_client() -> GeminiLLMClient:
    """Create a GeminiLLMClient for dashboard interactions."""
    return GeminiLLMClient(config_path=str(CONFIG_PATH))


def get_best_dataset() -> pd.DataFrame:
    """Return the richest currently available dataset for UI analytics."""
    for path in [ANNOTATED_DATASET_PATH, CLEAN_DATASET_PATH, RAW_DATASET_PATH]:
        df = read_parquet_safe(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def get_pipeline_stats() -> dict[str, Any]:
    """Compute compact pipeline stats for sidebar and analytics."""
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
    }


def compute_review_impact(threshold: float) -> tuple[int, float]:
    """Estimate queue size at the current confidence threshold."""
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
        "suggested_classes": st.session_state.get("current_classes", []),
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


def update_classes_with_llm(llm_client: GeminiLLMClient) -> None:
    """Refresh recommended classes from the current dataset summary."""
    dataset_df = get_best_dataset()
    topic = st.session_state.get("topic", "")
    current_classes = [
        label
        for label in st.session_state.get("current_classes", [])
        if label != st.session_state.get("review_label")
    ]
    if dataset_df.empty:
        summary = {
            "total_rows": 0,
            "source_distribution": {},
            "top_keywords_global": [],
            "pct_short_texts": 0.0,
            "pct_long_texts": 0.0,
            "current_topic": topic,
            "current_classes": current_classes,
        }
    else:
        summary = llm_client.build_dataset_summary(dataset_df)
        summary["current_topic"] = topic or summary.get("current_topic", "")
        summary["current_classes"] = current_classes

    spec = llm_client.generate_domain_spec(
        topic=topic or str(summary.get("current_topic", "")),
        dataset_summary=summary,
        current_classes=current_classes,
    )
    st.session_state["current_classes"] = spec.get("recommended_classes", current_classes)
    st.session_state["review_label"] = spec.get(
        "review_label",
        st.session_state.get("review_label", "other_or_offtopic"),
    )
    st.session_state["last_domain_spec"] = spec
    logger.info("Классы обновлены через LLM: {}", st.session_state["current_classes"])


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


def render_sidebar(llm_client: GeminiLLMClient) -> float:
    """Render sidebar controls and return the active threshold."""
    stats = get_pipeline_stats()
    cfg = load_config()
    st.sidebar.header("⚙️ Pipeline Settings")

    topic_value = st.sidebar.text_input(
        "Тема классификации",
        value=st.session_state.get("topic", ""),
        key="sidebar_topic",
    )
    st.session_state["topic"] = topic_value.strip()

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
    if (
        "classes_text" not in st.session_state
        or st.session_state.get("classes_text_last_synced")
        != "\n".join(current_classes_state)
    ):
        st.session_state["classes_text"] = "\n".join(current_classes_state)
        st.session_state["classes_text_last_synced"] = "\n".join(current_classes_state)

    st.sidebar.markdown("**Классы классификации:**")
    classes_input = st.sidebar.text_area(
        "Редактировать классы (каждый с новой строки):",
        value=st.session_state["classes_text"],
        height=150,
        help=(
            "Можно отредактировать предложенные LLM классы "
            "или написать свои с нуля"
        ),
        key="classes_text",
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
                cfg_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
                domain_cfg = cfg_data.setdefault("domain", {})
                domain_cfg["classes"] = new_classes
                CONFIG_PATH.write_text(
                    yaml.safe_dump(
                        cfg_data,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                st.session_state["current_classes"] = new_classes
                st.session_state["classes_text"] = "\n".join(new_classes)
                st.session_state["classes_text_last_synced"] = "\n".join(new_classes)
                st.sidebar.success(f"✅ Сохранено {len(new_classes)} классов")
                st.rerun()
            else:
                st.sidebar.error("Минимум 2 класса!")

    with col2:
        if st.button("↩️ Сбросить к дефолту", key="reset_sidebar_classes"):
            default_classes = [
                "navigation",
                "safety",
                "equipment",
                "weather",
                "licensing",
            ]
            st.session_state["classes_text"] = "\n".join(default_classes)
            st.session_state["classes_text_last_synced"] = "\n".join(default_classes)
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


def render_onboarding_tab(llm_client: GeminiLLMClient) -> None:
    """Render onboarding flow for topic and source discovery."""
    st.title("⛵ Smart Data Pipeline")

    if st.session_state.get("topic") and not st.session_state.get("editing_topic", False):
        st.subheader("Текущая конфигурация домена")
        st.write(f"**Тема:** {st.session_state.get('topic')}")
        st.write("**Классы:** " + ", ".join(st.session_state.get("current_classes", [])))
        if st.button("Изменить тему", key="change_topic_button"):
            st.session_state["editing_topic"] = True
            st.rerun()
        if st.session_state.get("selected_sources"):
            st.success(
                "Выбраны источники: "
                + ", ".join(st.session_state.get("selected_sources", []))
            )
        return

    st.subheader("Введите тему для классификации текстов")
    topic = st.text_input(
        "Тема пользователя",
        value=st.session_state.get("topic", ""),
        key="onboarding_topic_input",
    ).strip()
    if topic:
        st.session_state["topic"] = topic

    if st.button("🔍 Найти источники данных", key="find_sources_button"):
        with st.spinner("Gemini ищет источники..."):
            suggestion_payload = find_sources_with_llm(topic, llm_client)
        st.session_state["source_suggestions"] = suggestion_payload.get("sources", [])
        if suggestion_payload.get("suggested_classes"):
            st.session_state["current_classes"] = [
                str(item).strip()
                for item in suggestion_payload.get("suggested_classes", [])
                if str(item).strip()
            ]

    suggestions = st.session_state.get("source_suggestions", [])
    if suggestions:
        table_df = pd.DataFrame(suggestions).rename(
            columns={
                "name": "Источник",
                "type": "Тип",
                "license": "Лицензия",
                "estimated_rows": "Строк (ориентир)",
                "risk_level": "Риск скрапинга",
            }
        )
        visible_columns = [
            "Источник",
            "Тип",
            "Лицензия",
            "Строк (ориентир)",
            "Риск скрапинга",
        ]
        st.dataframe(
            style_source_table(table_df[visible_columns]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Выберите источники для использования**")
        selected_sources: list[str] = []
        for index, item in enumerate(suggestions):
            source_name = str(item.get("name", f"Источник {index + 1}"))
            if st.checkbox(source_name, key=f"source_checkbox_{index}", value=index < 3):
                selected_sources.append(source_name)
        if st.button("✅ Использовать выбранные источники", key="use_sources_button"):
            st.session_state["selected_sources"] = selected_sources
            st.session_state["editing_topic"] = False
            st.success(f"Сохранено источников: {len(selected_sources)}")


def render_hitl_tab(all_labels: list[str]) -> None:
    """Render manual review workflow for the review queue."""
    st.subheader("🔍 Проверка меток (HITL ★)")
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

    st.dataframe(pd.DataFrame(LICENSE_ROWS), use_container_width=True, hide_index=True)

    eda_path = pathlib.Path("reports/eda_report.html")
    if eda_path.exists():
        with open(eda_path, "rb") as report_file:
            st.download_button(
                label="📊 Скачать EDA отчёт (HTML)",
                data=report_file.read(),
                file_name="eda_report.html",
                mime="text/html",
            )
        st.info("💡 Для просмотра: откройте файл в браузере после скачивания")
    else:
        st.warning("EDA отчёт не найден. Запустите: python notebooks/export_eda.py")

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


def main() -> None:
    """Run the Streamlit HITL dashboard."""
    init_state()
    llm_client = build_llm_client()
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
