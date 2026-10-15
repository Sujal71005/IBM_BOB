"""
model_training.py
-----------------
Train and persist machine learning pipelines for LoanVision AI.

Three classifiers are trained:
  1. Logistic Regression   — interpretable linear baseline
  2. Random Forest         — ensemble; handles non-linearity well
  3. Gradient Boosting     — usually best AUC on tabular data; selected as final model

Best model selection criterion: ROC-AUC on held-out test set.
Rationale: The dataset is class-imbalanced (~31 % approved). Accuracy alone would
be misleading; AUC evaluates ranking quality across all decision thresholds, making
it a more reliable measure of discriminative ability for this task.

All sklearn Pipelines wrap (preprocessor → classifier) so that:
  - Imputation and scaling fit only on training data.
  - The same pipeline object is used for prediction — no leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
MODEL_PATH = MODELS_DIR / "loanvision_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"

RANDOM_STATE = 42


# ─── Build individual pipelines ──────────────────────────────────────────────────

def _build_pipelines(preprocessor):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced",
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                random_state=RANDOM_STATE, subsample=0.8,
            )),
        ]),
    }


# ─── Train all models ────────────────────────────────────────────────────────────

def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor=None,
    progress_callback=None,
) -> dict:
    """
    Train all pipelines; return dict {name: fitted_pipeline}.

    progress_callback(name: str, step: int, total: int) called after each model.
    """
    from src.data_processing import build_preprocessor

    if preprocessor is None:
        preprocessor = build_preprocessor()

    pipelines = _build_pipelines(preprocessor)
    fitted: dict = {}
    total = len(pipelines)

    for i, (name, pipe) in enumerate(pipelines.items(), 1):
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        if progress_callback:
            progress_callback(name, i, total)

    return fitted


# ─── Evaluate all models ─────────────────────────────────────────────────────────

def evaluate_all_models(fitted_pipelines: dict, X_test, y_test) -> list[dict]:
    """
    Evaluate all fitted pipelines on the test set.
    Returns list of metric dicts sorted by ROC-AUC desc.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    results = []
    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        y_proba = (
            pipe.predict_proba(X_test)[:, 1]
            if hasattr(pipe.named_steps["classifier"], "predict_proba")
            else None
        )
        cm = confusion_matrix(y_test, y_pred).tolist()
        row = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
            "confusion_matrix": cm,
        }
        results.append(row)

    results.sort(key=lambda r: (r["roc_auc"] or 0), reverse=True)
    return results


# ─── Select best model ───────────────────────────────────────────────────────────

def select_best_model(fitted_pipelines: dict, metrics: list[dict]) -> tuple[str, object]:
    """Return (best_model_name, best_pipeline) based on highest ROC-AUC."""
    best = metrics[0]  # already sorted desc by roc_auc
    return best["model"], fitted_pipelines[best["model"]]


# ─── Persist / load ──────────────────────────────────────────────────────────────

def save_best_pipeline(pipeline, name: str, metrics: list[dict]) -> None:
    """Persist the best pipeline and full metrics JSON."""
    import joblib

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    joblib.dump({"pipeline": pipeline, "model_name": name}, MODEL_PATH)

    payload = {
        "best_model": name,
        "selection_criterion": "ROC-AUC",
        "all_metrics": metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def load_best_pipeline() -> tuple[object, str]:
    """
    Load saved pipeline.
    Raises FileNotFoundError if not found — callers must handle this.
    """
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Go to 'Model Lab' and click 'Train Models' first."
        )
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["model_name"]


def load_metrics() -> dict | None:
    """Load saved metrics JSON; return None if not found."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ─── Full training run ───────────────────────────────────────────────────────────

def run_full_training(progress_callback=None) -> dict:
    """
    End-to-end: load data → split → train → evaluate → save.
    Returns metrics payload.
    """
    from src.data_processing import (
        build_preprocessor,
        get_X_y,
        load_clean_dataset,
        split_data,
    )

    df, _, _ = load_clean_dataset()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
    fitted = train_all_models(X_train, y_train, X_test, y_test, preprocessor, progress_callback)
    metrics = evaluate_all_models(fitted, X_test, y_test)
    best_name, best_pipe = select_best_model(fitted, metrics)
    save_best_pipeline(best_pipe, best_name, metrics)

    return {"best_model": best_name, "all_metrics": metrics}
