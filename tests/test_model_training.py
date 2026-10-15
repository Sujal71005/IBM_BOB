"""
Tests for model_training module.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import build_preprocessor, get_X_y, load_clean_dataset, split_data
from src.model_training import (
    evaluate_all_models,
    select_best_model,
    train_all_models,
)


@pytest.fixture(scope="module")
def trained_data():
    df, _, _ = load_clean_dataset()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def fitted_pipelines(trained_data):
    X_train, X_test, y_train, y_test = trained_data
    prep = build_preprocessor()
    fitted = train_all_models(X_train, y_train, X_test, y_test, preprocessor=prep)
    return fitted


def test_train_returns_three_models(fitted_pipelines):
    assert len(fitted_pipelines) == 3
    for name in ("Logistic Regression", "Random Forest", "Gradient Boosting"):
        assert name in fitted_pipelines


def test_evaluate_returns_metrics(fitted_pipelines, trained_data):
    _, X_test, _, y_test = trained_data
    metrics = evaluate_all_models(fitted_pipelines, X_test, y_test)
    assert len(metrics) == 3
    for m in metrics:
        for key in ("model", "accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in m


def test_metrics_in_valid_range(fitted_pipelines, trained_data):
    _, X_test, _, y_test = trained_data
    metrics = evaluate_all_models(fitted_pipelines, X_test, y_test)
    for m in metrics:
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0


def test_select_best_model(fitted_pipelines, trained_data):
    _, X_test, _, y_test = trained_data
    metrics = evaluate_all_models(fitted_pipelines, X_test, y_test)
    best_name, best_pipe = select_best_model(fitted_pipelines, metrics)
    assert best_name in fitted_pipelines
    assert best_pipe is fitted_pipelines[best_name]


def test_metrics_sorted_by_roc_auc(fitted_pipelines, trained_data):
    _, X_test, _, y_test = trained_data
    metrics = evaluate_all_models(fitted_pipelines, X_test, y_test)
    aucs = [m["roc_auc"] for m in metrics]
    assert aucs == sorted(aucs, reverse=True)
