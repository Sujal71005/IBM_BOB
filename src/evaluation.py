"""
evaluation.py
-------------
Model evaluation utilities for LoanVision AI.
All evaluations use held-out test data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cv_scores(pipeline, X, y, cv: int = 5) -> dict:
    """
    Compute cross-validated AUC, accuracy, and F1 scores.
    Uses StratifiedKFold so each fold preserves class ratio.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {}
    for metric in ("roc_auc", "accuracy", "f1"):
        s = cross_val_score(pipeline, X, y, cv=skf, scoring=metric)
        scores[metric] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "folds": [round(float(x), 4) for x in s],
        }
    return scores


def plot_confusion_matrix(cm: list[list[int]], model_name: str):
    """Return a Plotly heatmap figure for the confusion matrix."""
    import plotly.graph_objects as go

    labels = ["Rejected (0)", "Approved (1)"]
    z = cm
    text = [[str(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_roc_curves(fitted_pipelines: dict, X_test, y_test):
    """Return a Plotly figure overlaying ROC curves for all models."""
    import plotly.graph_objects as go
    from sklearn.metrics import roc_auc_score, roc_curve

    fig = go.Figure()
    for name, pipe in fitted_pipelines.items():
        if hasattr(pipe.named_steps["classifier"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc:.3f})",
            ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random",
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_metrics_comparison(metrics: list[dict]):
    """Return a grouped bar chart comparing model metrics."""
    import plotly.graph_objects as go

    model_names = [m["model"] for m in metrics]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = ["#3b82d4", "#7c5cd8", "#10b981", "#f59e0b", "#ef4444"]

    fig = go.Figure()
    for key, label, color in zip(metric_keys, metric_labels, colors):
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=[m.get(key) or 0 for m in metrics],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        title="Model Comparison — Key Metrics",
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis_title="Model",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
