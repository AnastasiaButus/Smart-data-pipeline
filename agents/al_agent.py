"""Active Learning agent for step 4.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from core.context_memory import ContextMemory


class ActiveLearningAgent:
    """Run uncertainty-based active learning cycles on text labels."""

    _DEFAULT_CONFIG = {
        "strategy": "entropy",
        "initial_size": 50,
        "batch_size": 20,
        "n_iterations": 5,
        "compare_strategies": True,
    }

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load config and AL parameters without building the model eagerly."""
        self._config_path = Path(config_path)
        self._project_root = self._config_path.parent
        self._cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        self._al_cfg = {
            **self._DEFAULT_CONFIG,
            **(self._cfg.get("active_learning", {}) or {}),
        }
        self._data_cfg = self._cfg.get("data", {}) or {}
        self._review_label = str(
            self._cfg.get("domain", {}).get("review_label", "other_or_offtopic")
        ).strip() or "other_or_offtopic"

        self._strategy = str(self._al_cfg.get("strategy", "entropy")).strip() or "entropy"
        self._initial_size = int(self._al_cfg.get("initial_size", 50))
        self._batch_size = int(self._al_cfg.get("batch_size", 20))
        self._n_iterations = int(self._al_cfg.get("n_iterations", 5))
        self._compare_strategies = bool(self._al_cfg.get("compare_strategies", True))

        self._labeled_path = self._project_root / str(
            self._data_cfg.get("labeled_path", "data/labeled")
        )
        self._review_queue_path = self._project_root / str(
            self._data_cfg.get("review_queue_path", "data/review_queue.csv")
        )
        self._reports_path = self._project_root / "reports"
        self._context_memory_path = self._reports_path / "context_memory.json"

        self._model: Pipeline | None = None
        self._classes_: list[str] = []
        self._last_history: list[dict[str, Any]] = []
        self._last_strategy_comparison: dict[str, list[dict[str, Any]]] = {}
        self._last_summary: dict[str, Any] = {}
        logger.info(
            "ActiveLearningAgent initialized. strategy={}, initial_size={}, batch_size={}, n_iterations={}",
            self._strategy,
            self._initial_size,
            self._batch_size,
            self._n_iterations,
        )

    def fit(self, labeled_df: pd.DataFrame) -> None:
        """Fit a TF-IDF + LogisticRegression model on labeled texts."""
        prepared = self._prepare_labeled_df(labeled_df)
        if prepared.empty:
            raise ValueError("No labeled data available for ActiveLearningAgent.fit()")

        x_train = prepared["text"].fillna("").astype(str)
        y_train = prepared["label"].astype(str)
        self._classes_ = sorted(y_train.unique().tolist())

        if len(self._classes_) < 2:
            logger.warning("Only one class available for fit(). Falling back to DummyClassifier.")
            self._model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=5000)),
                    ("clf", DummyClassifier(strategy="most_frequent")),
                ]
            )
        else:
            self._model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=5000)),
                    ("clf", LogisticRegression(max_iter=1000)),
                ]
            )

        self._model.fit(x_train, y_train)
        train_pred = self._model.predict(x_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        logger.info(
            "Active Learning model fitted on {} rows with train accuracy={:.4f}",
            len(prepared),
            train_accuracy,
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probability estimates with shape ``(n_samples, n_classes)``."""
        if self._model is None:
            raise ValueError("Model is not fitted. Call fit() before predict_proba().")

        clean_texts = [str(text) for text in texts]
        proba = self._model.predict_proba(clean_texts)
        return np.asarray(proba)

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str | None = None,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Select the most informative rows from the pool by the chosen strategy."""
        prepared_pool = self._prepare_pool_df(pool_df)
        if prepared_pool.empty:
            return prepared_pool

        active_strategy = self._resolve_strategy(strategy)
        n_samples = max(1, int(n or self._batch_size))
        n_samples = min(n_samples, len(prepared_pool))

        if active_strategy == "random":
            selected = prepared_pool.sample(n=n_samples, random_state=42)
            return selected.copy()

        proba = self.predict_proba(prepared_pool["text"].fillna("").astype(str).tolist())

        if active_strategy == "entropy":
            scores = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            order = np.argsort(scores)[::-1][:n_samples]
        else:
            sorted_proba = np.sort(proba, axis=1)
            if sorted_proba.shape[1] == 1:
                margins = np.zeros(sorted_proba.shape[0])
            else:
                margins = sorted_proba[:, -1] - sorted_proba[:, -2]
            order = np.argsort(margins)[:n_samples]

        return prepared_pool.iloc[order].copy()

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Evaluate the classifier and return accuracy / F1 metrics."""
        prepared = self._prepare_labeled_df(labeled_df)
        if prepared.empty:
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "f1_per_class": {},
                "n_train": 0,
                "n_test": 0,
            }

        if test_df is None:
            if len(prepared) < 5:
                train_df = prepared
                holdout_df = prepared
            else:
                label_counts = prepared["label"].value_counts()
                estimated_test_size = max(1, int(round(len(prepared) * 0.2)))
                stratify = (
                    prepared["label"]
                    if (
                        label_counts.min() >= 2
                        and label_counts.size >= 2
                        and estimated_test_size >= label_counts.size
                    )
                    else None
                )
                train_df, holdout_df = train_test_split(
                    prepared,
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify,
                )
        else:
            train_df = prepared
            holdout_df = self._prepare_labeled_df(test_df)

        self.fit(train_df)
        y_true = holdout_df["label"].astype(str)
        y_pred = self._model.predict(holdout_df["text"].fillna("").astype(str))
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

        f1_per_class = {
            label: round(
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=[label],
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            )
            for label in labels
        }
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_per_class": f1_per_class,
            "n_train": int(len(train_df)),
            "n_test": int(len(holdout_df)),
        }
        return metrics

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        strategy: str | None = None,
        n_iterations: int | None = None,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run an active learning loop and return a compact metric history."""
        prepared_labeled = self._prepare_labeled_df(labeled_df)
        prepared_pool = self._prepare_pool_df(pool_df)
        active_strategy = self._resolve_strategy(strategy)
        total_iterations = int(n_iterations or self._n_iterations)
        step_batch_size = int(batch_size or self._batch_size)

        if prepared_labeled.empty:
            logger.warning("run_cycle() received no labeled data")
            return []

        if len(prepared_labeled) < self._initial_size:
            logger.warning(
                "Available labeled data ({}) is smaller than initial_size ({}). Using all rows.",
                len(prepared_labeled),
                self._initial_size,
            )
            current_labeled = prepared_labeled.copy()
        else:
            current_labeled = self._sample_initial_labeled(prepared_labeled, self._initial_size)

        current_pool = prepared_pool.copy()
        history: list[dict[str, Any]] = []

        for iteration in range(1, total_iterations + 1):
            self.fit(current_labeled)
            metrics = self.evaluate(current_labeled)
            history.append(
                {
                    "iteration": iteration,
                    "n_labeled": int(len(current_labeled)),
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "strategy": active_strategy,
                }
            )

            if current_pool.empty:
                logger.warning("Pool exhausted at iteration {}", iteration)
                break

            queried = self.query(current_pool, strategy=active_strategy, n=step_batch_size)
            if queried.empty:
                logger.warning("No rows were queried at iteration {}", iteration)
                break

            current_labeled = pd.concat(
                [current_labeled, self._prepare_pool_df(queried)],
                ignore_index=True,
            )
            current_pool = current_pool.drop(index=queried.index, errors="ignore").reset_index(drop=True)

        self._last_history = history
        return history

    def report(self, history: list[dict[str, Any]]) -> None:
        """Save an interactive learning curve and try to export a static PNG."""
        if not history:
            logger.warning("report() skipped because history is empty")
            return

        history_df = pd.DataFrame(history)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=history_df["n_labeled"],
                y=history_df["accuracy"],
                mode="lines+markers",
                name="accuracy",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=history_df["n_labeled"],
                y=history_df["f1_macro"],
                mode="lines+markers",
                name="f1_macro",
            )
        )
        figure.update_layout(
            title="Learning curve",
            xaxis_title="n_labeled",
            yaxis_title="score",
            template="plotly_white",
        )

        self._reports_path.mkdir(parents=True, exist_ok=True)
        html_path = self._reports_path / "learning_curve.html"
        png_path = self._reports_path / "learning_curve.png"
        figure.write_html(str(html_path), include_plotlyjs="cdn")
        logger.info("Learning curve saved to {}", html_path)

        try:
            figure.write_image(str(png_path))
            logger.info("Learning curve PNG saved to {}", png_path)
        except Exception as exc:
            logger.warning("PNG export skipped (kaleido optional): {}", exc)

    def compare_strategies(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run AL cycles for entropy, margin, and random and save a comparison plot."""
        comparisons = {
            strategy: self.run_cycle(labeled_df, pool_df, strategy=strategy)
            for strategy in ["entropy", "margin", "random"]
        }

        figure = go.Figure()
        for strategy, history in comparisons.items():
            if not history:
                continue
            history_df = pd.DataFrame(history)
            figure.add_trace(
                go.Scatter(
                    x=history_df["n_labeled"],
                    y=history_df["f1_macro"],
                    mode="lines+markers",
                    name=strategy,
                )
            )
        figure.update_layout(
            title="Strategy comparison",
            xaxis_title="n_labeled",
            yaxis_title="f1_macro",
            template="plotly_white",
        )
        comparison_path = self._reports_path / "strategy_comparison.html"
        self._reports_path.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(comparison_path), include_plotlyjs="cdn")
        logger.info("Strategy comparison saved to {}", comparison_path)

        self._last_strategy_comparison = comparisons
        return comparisons

    def summary(self) -> dict[str, Any]:
        """Return a compact AL summary without any raw texts."""
        return dict(self._last_summary)

    def run(
        self,
        labeled_df: pd.DataFrame | None = None,
        pool_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run the configured AL cycle, compare strategies, and update memory."""
        if labeled_df is None:
            annotated_path = self._labeled_path / "annotated.parquet"
            if not annotated_path.exists():
                raise FileNotFoundError(f"Annotated dataset not found: {annotated_path}")
            labeled_df = pd.read_parquet(annotated_path)
            labeled_df = labeled_df.loc[
                labeled_df["label"].astype(str).ne("unlabeled")
                & labeled_df["label"].astype(str).ne(self._review_label)
            ].copy()

        if pool_df is None:
            if not self._review_queue_path.exists():
                raise FileNotFoundError(f"Review queue not found: {self._review_queue_path}")
            pool_df = pd.read_csv(self._review_queue_path)

        cycle_history = self.run_cycle(
            labeled_df=labeled_df,
            pool_df=pool_df,
            strategy=self._strategy,
            n_iterations=self._n_iterations,
            batch_size=self._batch_size,
        )
        comparison = (
            self.compare_strategies(labeled_df, pool_df)
            if self._compare_strategies
            else {self._strategy: cycle_history}
        )
        self.report(cycle_history)

        best_strategy = self._strategy
        best_f1 = -1.0
        for strategy, history in comparison.items():
            if history and history[-1]["f1_macro"] > best_f1:
                best_strategy = strategy
                best_f1 = history[-1]["f1_macro"]

        final_metrics = cycle_history[-1] if cycle_history else {
            "accuracy": 0.0,
            "f1_macro": 0.0,
        }
        self._last_summary = {
            "strategy": self._strategy,
            "n_iterations": len(cycle_history),
            "final_accuracy": round(float(final_metrics.get("accuracy", 0.0)), 4),
            "final_f1": round(float(final_metrics.get("f1_macro", 0.0)), 4),
            "best_strategy": best_strategy,
        }
        ContextMemory(path=str(self._context_memory_path)).update(
            step="4.1",
            status="done",
            metrics=self._last_summary,
            notes="AL cycle completed",
        )
        return self.summary()

    def _prepare_labeled_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a labeled dataframe for AL fitting and evaluation."""
        working_df = df.copy()
        if "text" not in working_df.columns or "label" not in working_df.columns:
            raise ValueError("AL dataframes must contain 'text' and 'label' columns")
        working_df["text"] = working_df["text"].fillna("").astype(str)
        working_df["label"] = working_df["label"].fillna("").astype(str)
        working_df = working_df.loc[
            working_df["text"].str.strip().ne("")
            & working_df["label"].str.strip().ne("")
        ].reset_index(drop=True)
        return working_df

    def _prepare_pool_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a pool dataframe and resolve labels from corrected values when present."""
        working_df = df.copy()
        if "text" not in working_df.columns:
            raise ValueError("Pool dataframe must contain a 'text' column")
        working_df["text"] = working_df["text"].fillna("").astype(str)
        if "corrected_label" in working_df.columns:
            corrected = working_df["corrected_label"].fillna("").astype(str).str.strip()
            suggested = (
                working_df["label"].fillna("").astype(str)
                if "label" in working_df.columns
                else ""
            )
            working_df["label"] = np.where(corrected.ne(""), corrected, suggested)
        elif "label" not in working_df.columns:
            working_df["label"] = ""
        else:
            working_df["label"] = working_df["label"].fillna("").astype(str)

        working_df = working_df.loc[working_df["text"].str.strip().ne("")].copy()
        if "id" not in working_df.columns:
            working_df["id"] = [str(index + 1) for index in range(len(working_df))]
        return working_df

    def _sample_initial_labeled(self, labeled_df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample an initial labeled subset while keeping class diversity when possible."""
        per_class = labeled_df.groupby("label", group_keys=False).head(1)
        remaining_needed = max(n - len(per_class), 0)
        if remaining_needed == 0:
            return per_class.sample(frac=1.0, random_state=42).reset_index(drop=True)

        leftovers = labeled_df.drop(index=per_class.index, errors="ignore")
        if leftovers.empty:
            return per_class.reset_index(drop=True)

        extra = leftovers.sample(
            n=min(remaining_needed, len(leftovers)),
            random_state=42,
        )
        return pd.concat([per_class, extra], ignore_index=True)

    def _resolve_strategy(self, strategy: str | None) -> str:
        """Resolve and validate the active learning strategy."""
        active_strategy = str(strategy or self._strategy).strip().lower()
        if active_strategy not in {"entropy", "margin", "random"}:
            logger.warning("Unknown strategy '{}', falling back to entropy", active_strategy)
            return "entropy"
        return active_strategy
