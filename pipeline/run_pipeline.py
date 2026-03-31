"""Prefect orchestration for the full smart-data-pipeline."""

from __future__ import annotations

import sys
import os
import pathlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

# Добавляем корень проекта в Python path
# чтобы находить core/, agents/ из любой папки
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")

from prefect import flow, task

load_dotenv()


def _load_config() -> dict[str, Any]:
    """Load project config for topic synchronization in the pipeline."""
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _write_config(config: dict[str, Any]) -> None:
    """Persist project config using UTF-8."""
    CONFIG_PATH.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _persist_pipeline_topic(
    *,
    topic: str | None = None,
    normalized_topic: str | None = None,
) -> None:
    """Persist current UI topic and last-generated data topic."""
    config = _load_config()
    domain_cfg = config.setdefault("domain", {})
    if topic is not None:
        domain_cfg["topic"] = topic
    if normalized_topic is not None:
        domain_cfg["normalized_topic"] = normalized_topic
    _write_config(config)


@task(name="collect_data")
def collect() -> pd.DataFrame:
    """Collect raw data through the existing data collection agent."""
    from agents.data_collection_agent import DataCollectionAgent

    agent = DataCollectionAgent()
    return agent.run()


@task(name="clean_data")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe with the existing quality agent."""
    from agents.data_quality_agent import DataQualityAgent

    agent = DataQualityAgent()
    return agent.run(df)


@task(name="annotate_data")
def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-label the cleaned dataframe and create the review queue."""
    from agents.annotation_agent import AnnotationAgent

    agent = AnnotationAgent()
    return agent.run(df)


@task(name="human_review")
def human_review(df_confident: pd.DataFrame, df_review: pd.DataFrame | None) -> pd.DataFrame:
    """Merge manually corrected review-queue rows back into the confident set."""
    from core.context_memory import ContextMemory

    review_path = Path("data/review_queue.csv")
    if review_path.exists():
        review_queue = pd.read_csv(review_path)
        corrected = review_queue.loc[
            review_queue["corrected_label"].notna()
            & review_queue["corrected_label"].astype(str).ne("")
        ].copy()
        logger.info("HITL: {} примеров проверено человеком", len(corrected))

        if len(corrected) > 0:
            corrected_df = corrected[
                ["id", "text", "corrected_label", "source"]
            ].rename(columns={"corrected_label": "label"})
            corrected_df["confidence"] = 1.0
            corrected_df["label_source"] = "hitl"
            merged = pd.concat([df_confident, corrected_df], ignore_index=True)
            ContextMemory().update(
                step="3.2",
                status="done",
                metrics={
                    "reviewed_examples": int(len(corrected)),
                    "confident_rows": int(len(df_confident)),
                    "rows_after_merge": int(len(merged)),
                },
                notes="HITL corrections merged into training set",
            )
            return merged

    return df_confident


@task(name="active_learning")
def active_learn(df: pd.DataFrame) -> dict[str, Any]:
    """Run active learning on the reviewed dataset."""
    from agents.al_agent import ActiveLearningAgent

    agent = ActiveLearningAgent()
    return agent.run(df)


@task(name="train_model")
def train(df: pd.DataFrame) -> dict[str, Any]:
    """Train the final model harness on the reviewed dataset."""
    from core.model_wrapper import ModelWrapper

    wrapper = ModelWrapper()
    try:
        return wrapper.fit(df)
    except ValueError as exc:
        message = str(exc)
        recoverable_markers = [
            "No labeled rows available for model training",
            "Need at least 2 classes with >=2 samples for training",
        ]
        if any(marker in message for marker in recoverable_markers):
            logger.warning(
                "Training skipped because the reviewed dataset is not ready yet: {}",
                message,
            )
            return {
                "skipped": True,
                "reason": message,
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "f1_per_class": {},
                "n_train": 0,
                "n_test": 0,
            }
        raise


@flow(
    name="smart-data-pipeline",
    description="End-to-end text classification pipeline",
)
def data_pipeline(
    skip_hitl: bool = False,
    skip_al: bool = False,
    topic: str | None = None,
) -> dict[str, Any]:
    """Run the full smart-data-pipeline with optional HITL and AL skips."""
    from core.context_memory import ContextMemory

    logger.info("=== Smart Data Pipeline START ===")
    config = _load_config()
    effective_topic = str(
        topic
        or config.get("domain", {}).get("topic", "sailing and yacht navigation")
    ).strip()
    if effective_topic:
        logger.info("Pipeline topic: {}", effective_topic)
        _persist_pipeline_topic(topic=effective_topic)

    raw_df = collect()
    clean_df = clean(raw_df)
    annotation_result = annotate(clean_df)

    if skip_hitl:
        logger.warning(
            "⚠️ HITL пропущен! Модель обучается только на авторазметке."
        )
        reviewed_df = annotation_result
    else:
        reviewed_df = human_review(annotation_result, None)

    if skip_al:
        logger.warning("⚠️ Active Learning пропущен!")
        al_result: dict[str, Any] = {"skipped": True}
    else:
        al_result = active_learn(reviewed_df)
        logger.info("AL summary: {}", al_result)

    _persist_pipeline_topic(normalized_topic=effective_topic)
    metrics = train(reviewed_df)

    logger.info("=== Pipeline COMPLETE ===")
    if metrics.get("skipped"):
        logger.warning("Training skipped: {}", metrics.get("reason", "unknown reason"))
    else:
        logger.info("Accuracy: {:.3f}", metrics["accuracy"])
        logger.info("F1 macro: {:.3f}", metrics["f1_macro"])

    ContextMemory().update(
        step="6.1",
        status="done",
        metrics={**metrics, "topic": effective_topic},
        notes=(
            f"Pipeline data refresh completed for topic: {effective_topic}; "
            f"training skipped: {metrics.get('reason')}"
            if metrics.get("skipped")
            else f"Full pipeline completed for topic: {effective_topic}"
        ),
    )
    return metrics


if __name__ == "__main__":
    results = data_pipeline()
    if results.get("skipped"):
        logger.warning("Pipeline data refresh completed, but training was skipped.")
        logger.warning("Reason: {}", results.get("reason", "unknown reason"))
    else:
        logger.success("✅ Pipeline done!")
        logger.success("Accuracy: {:.3f}", results["accuracy"])
        logger.success("F1 macro: {:.3f}", results["f1_macro"])
