"""Model harness for training, evaluation, prediction, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class ModelWrapper:
    """Lazy model harness for sklearn baseline training and inference."""

    _DEFAULT_MODEL_TYPE = "sklearn"
    _DEFAULT_TEST_SIZE = 0.2
    _DEFAULT_RANDOM_STATE = 42

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load config and initialize lazy model state."""
        self._config_path = Path(config_path)
        self._project_root = self._config_path.parent
        self._cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        self._model_cfg = self._cfg.get("model", {}) or {}
        self.model_type = str(
            self._model_cfg.get("type", self._DEFAULT_MODEL_TYPE)
        ).strip() or self._DEFAULT_MODEL_TYPE
        self._exclude_offtopic = bool(self._model_cfg.get("exclude_offtopic", True))
        self._test_size = float(
            self._model_cfg.get("test_size", self._DEFAULT_TEST_SIZE)
        )
        self._random_state = int(
            self._model_cfg.get("random_state", self._DEFAULT_RANDOM_STATE)
        )
        self._review_label = str(
            self._cfg.get("domain", {}).get("review_label", "other_or_offtopic")
        ).strip() or "other_or_offtopic"

        self._model_path = self._project_root / "models" / "classifier.pkl"
        self._label_encoder_path = self._project_root / "models" / "label_encoder.pkl"
        self._metrics_path = self._project_root / "reports" / "model_metrics.json"

        self._model: Pipeline | None = None
        self._label_encoder: LabelEncoder | None = None
        self._classes: list[str] = []
        self._last_metrics: dict[str, Any] = {}
        logger.info("ModelWrapper initialized with model_type={}", self.model_type)

    def _build_sklearn_model(self) -> Pipeline:
        """Build the sklearn text classification pipeline."""
        tfidf_cfg = self._model_cfg.get("tfidf", {}) or {}
        logreg_cfg = self._model_cfg.get("logreg", {}) or {}
        ngram_range_raw = tfidf_cfg.get("ngram_range", [1, 2])
        if isinstance(ngram_range_raw, list | tuple) and len(ngram_range_raw) == 2:
            ngram_range = (int(ngram_range_raw[0]), int(ngram_range_raw[1]))
        else:
            ngram_range = (1, 2)

        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=int(tfidf_cfg.get("max_features", 10000)),
                        ngram_range=ngram_range,
                        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(logreg_cfg.get("max_iter", 1000)),
                        class_weight=str(
                            logreg_cfg.get("class_weight", "balanced")
                        ).strip()
                        or "balanced",
                        C=float(logreg_cfg.get("C", 1.0)),
                        random_state=self._random_state,
                    ),
                ),
            ]
        )

    @staticmethod
    def _build_distilbert_stub() -> None:
        """Raise the planned DistilBERT upgrade stub."""
        logger.info("DistilBERT stub - not implemented")
        logger.info(
            "To enable: set model.type=distilbert and pip install transformers torch"
        )
        raise NotImplementedError(
            "DistilBERT upgrade planned. Use model.type=sklearn for now."
        )

    def fit(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
    ) -> dict[str, Any]:
        """Train the configured model, evaluate it, and persist artifacts."""
        working_df = df.copy()
        if text_col not in working_df.columns or label_col not in working_df.columns:
            raise ValueError(
                f"Training dataframe must contain '{text_col}' and '{label_col}' columns"
            )

        working_df[text_col] = working_df[text_col].fillna("").astype(str)
        working_df[label_col] = working_df[label_col].fillna("").astype(str)
        working_df = working_df.loc[working_df[label_col].ne("unlabeled")].copy()
        if self._exclude_offtopic:
            working_df = working_df.loc[
                working_df[label_col].ne(self._review_label)
            ].copy()
        working_df = working_df.loc[working_df[text_col].str.strip().ne("")].copy()
        working_df = working_df.reset_index(drop=True)

        if working_df.empty:
            raise ValueError("No labeled rows available for model training")

        label_counts = working_df[label_col].value_counts()
        valid_labels = label_counts[label_counts >= 2].index.tolist()
        if len(valid_labels) >= 2 and len(valid_labels) < label_counts.size:
            logger.warning(
                "Dropping classes with <2 samples before stratified split: {}",
                sorted(set(label_counts.index) - set(valid_labels)),
            )
            working_df = working_df.loc[working_df[label_col].isin(valid_labels)].copy()
            label_counts = working_df[label_col].value_counts()

        if label_counts.size < 2:
            raise ValueError("Need at least 2 classes with >=2 samples for training")

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(working_df[label_col])
        self._classes = [str(label) for label in self._label_encoder.classes_.tolist()]

        if self.model_type == "distilbert":
            self._build_distilbert_stub()

        self._model = self._build_sklearn_model()

        X = working_df[text_col].tolist()
        y = y_encoded
        stratify_target: np.ndarray | None = y if len(np.unique(y)) > 1 else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self._test_size,
                random_state=self._random_state,
                stratify=stratify_target,
            )
        except ValueError as exc:
            logger.warning(
                "Falling back to non-stratified split due to data constraints: {}",
                exc,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self._test_size,
                random_state=self._random_state,
                stratify=None,
            )

        self._model.fit(X_train, y_train)
        train_accuracy = float(self._model.score(X_train, y_train))
        logger.info("Model trained. train_accuracy={:.4f}", train_accuracy)

        metrics = self.evaluate(X_test, y_test)
        metrics["train_accuracy"] = round(train_accuracy, 4)
        metrics["n_train"] = int(len(X_train))

        self.save()
        self._label_encoder_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._label_encoder, self._label_encoder_path)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._last_metrics = metrics
        logger.info("Model artifacts saved to {}", self._model_path.parent)
        return metrics

    def predict(self, texts: list[str]) -> list[str]:
        """Predict labels for a list of texts."""
        model, encoder = self._ensure_loaded()
        normalized_texts = [str(text) for text in texts]
        predictions = model.predict(normalized_texts)
        decoded = encoder.inverse_transform(np.asarray(predictions, dtype=int))
        return [str(label) for label in decoded.tolist()]

    def predict_proba(self, texts: list[str]) -> dict[str, Any]:
        """Return class labels and probability matrix for the supplied texts."""
        model, encoder = self._ensure_loaded()
        normalized_texts = [str(text) for text in texts]
        probas = model.predict_proba(normalized_texts)
        return {
            "labels": [str(label) for label in encoder.classes_.tolist()],
            "probas": probas.tolist(),
        }

    def evaluate(self, X_test: list[str], y_test: Any) -> dict[str, Any]:
        """Evaluate the current model on a held-out dataset."""
        model, encoder = self._ensure_loaded()
        X_eval = [str(text) for text in list(X_test)]
        y_true = np.asarray(y_test)
        predicted_encoded = model.predict(X_eval)
        y_true_labels = encoder.inverse_transform(y_true.astype(int))
        y_pred_labels = encoder.inverse_transform(predicted_encoded.astype(int))

        accuracy = round(float(accuracy_score(y_true_labels, y_pred_labels)), 4)
        f1_macro = round(
            float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
            4,
        )
        f1_weighted = round(
            float(
                f1_score(
                    y_true_labels,
                    y_pred_labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        )

        labels = [str(label) for label in encoder.classes_.tolist()]
        per_class_scores = f1_score(
            y_true_labels,
            y_pred_labels,
            labels=labels,
            average=None,
            zero_division=0,
        )
        classification = classification_report(
            y_true_labels,
            y_pred_labels,
            labels=labels,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_per_class": {
                label: round(float(score), 4)
                for label, score in zip(labels, per_class_scores)
            },
            "classification_report": classification,
            "n_test": int(len(X_eval)),
        }

    def explain(self, texts: list[str], n_features: int = 10) -> list[dict[str, Any]]:
        """Explain predictions with top contributing TF-IDF features."""
        try:
            model, encoder = self._ensure_loaded()
            vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
            classifier: LogisticRegression = model.named_steps["clf"]
            features = vectorizer.get_feature_names_out()
            predictions = self.predict(texts)
            transformed = vectorizer.transform([str(text) for text in texts])
            explanations: list[dict[str, Any]] = []

            for index, text in enumerate(texts):
                predicted_label = predictions[index]
                class_idx = int(encoder.transform([predicted_label])[0])
                if classifier.coef_.shape[0] == 1 and len(encoder.classes_) == 2:
                    coef = (
                        classifier.coef_[0]
                        if class_idx == 1
                        else -classifier.coef_[0]
                    )
                else:
                    coef = classifier.coef_[class_idx]

                row = transformed[index]
                contributions = row.multiply(coef).toarray().ravel()
                top_indices = np.argsort(contributions)[::-1]
                top_features: list[tuple[str, float]] = []
                for feature_index in top_indices:
                    weight = float(contributions[feature_index])
                    if weight <= 0:
                        continue
                    top_features.append(
                        (str(features[feature_index]), round(weight, 6))
                    )
                    if len(top_features) >= n_features:
                        break

                explanations.append(
                    {
                        "text": str(text),
                        "label": predicted_label,
                        "top_features": top_features,
                    }
                )
            return explanations
        except Exception as exc:
            logger.warning("Feature explanation fallback used: {}", exc)
            fallback_labels = self.predict(texts) if self._can_predict() else ["" for _ in texts]
            return [
                {
                    "text": str(text),
                    "label": str(label),
                    "top_features": [],
                }
                for text, label in zip(texts, fallback_labels)
            ]

    def save(self, path: str = "models/classifier.pkl") -> None:
        """Persist the fitted model to disk with joblib."""
        if self._model is None:
            raise ValueError("Model is not trained yet")
        target_path = self._project_root / path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, target_path)
        logger.info("Model saved to {}", target_path)

    def load(self, path: str = "models/classifier.pkl") -> None:
        """Load the fitted model and label encoder from disk."""
        target_path = self._project_root / path
        encoder_path = self._label_encoder_path
        try:
            self._model = joblib.load(target_path)
            self._label_encoder = joblib.load(encoder_path)
            self._classes = [
                str(label) for label in self._label_encoder.classes_.tolist()
            ]
            logger.info("Model loaded from {}", target_path)
        except Exception as exc:
            logger.error("Failed to load model artifacts: {}", exc)
            raise

    def summary(self) -> dict[str, Any]:
        """Return a compact model summary without raw texts."""
        return {
            "model_type": self.model_type,
            "n_classes": len(self._classes),
            "classes": list(self._classes),
            "accuracy": self._last_metrics.get("accuracy", 0.0),
            "f1_macro": self._last_metrics.get("f1_macro", 0.0),
            "trained": self._model is not None and self._label_encoder is not None,
        }

    def _ensure_loaded(self) -> tuple[Pipeline, LabelEncoder]:
        """Ensure model artifacts are loaded before inference or evaluation."""
        if self._model is None or self._label_encoder is None:
            self.load()
        if self._model is None or self._label_encoder is None:
            raise ValueError("Model artifacts are not available")
        return self._model, self._label_encoder

    def _can_predict(self) -> bool:
        """Return whether the wrapper currently has enough artifacts for prediction."""
        return self._model is not None and self._label_encoder is not None
