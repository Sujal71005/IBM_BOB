"""
explainability.py
-----------------
Model interpretation utilities for LoanVision AI.

Methods used:
  - Global: feature importances from tree-based models; coefficient magnitudes
    from Logistic Regression; permutation importance as a model-agnostic fallback.
  - Local: for a single prediction, SHAP-style approximation via perturbation
    of each feature (if SHAP unavailable), or coefficient × feature value for
    linear models.

Limitations:
  - Importances from one-hot-encoded pipelines are reported per *original* feature
    by summing child importances — avoids misleading split of categorical features.
  - Permutation importance is computed on held-out test data, not training data.
  - Associations are correlational, not causal.
  - The model may reflect biases in historical labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    get_X_y,
    load_clean_dataset,
    split_data,
)


def _get_feature_names_from_pipeline(pipeline) -> list[str]:
    """Extract the full list of encoded feature names from a fitted ColumnTransformer."""
    ct = pipeline.named_steps["preprocessor"]
    names = []
    for tname, trans, cols in ct.transformers_:
        if tname == "num":
            names.extend(cols)
        elif tname == "cat":
            enc = trans.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols).tolist())
    return names


def _aggregate_to_original(importances: np.ndarray, encoded_names: list[str]) -> dict[str, float]:
    """
    Map encoded importances back to original feature names by summing
    one-hot child columns.
    """
    result: dict[str, float] = {}
    for feat, imp in zip(encoded_names, importances):
        # One-hot columns are named like "Employment_Status_Salaried"
        original = feat.split("_")[0]
        # Better: match by prefix against ALL_FEATURES
        matched = None
        for f in ALL_FEATURES:
            if feat == f or feat.startswith(f + "_"):
                matched = f
                break
        if matched is None:
            matched = feat
        result[matched] = result.get(matched, 0.0) + float(imp)
    return result


def get_global_feature_importance(pipeline) -> pd.DataFrame:
    """
    Return a DataFrame with columns [feature, importance, direction].
    direction: +1 positive association with approval, -1 negative, 0 unknown.
    """
    clf = pipeline.named_steps["classifier"]
    encoded_names = _get_feature_names_from_pipeline(pipeline)
    clf_name = type(clf).__name__

    if hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
        agg = _aggregate_to_original(raw, encoded_names)
        df = pd.DataFrame(agg.items(), columns=["feature", "importance"])
        df["direction"] = 0  # tree importances are unsigned
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        agg_pos = _aggregate_to_original(np.maximum(coefs, 0), encoded_names)
        agg_neg = _aggregate_to_original(np.minimum(coefs, 0), encoded_names)
        agg_abs = _aggregate_to_original(np.abs(coefs), encoded_names)
        rows = []
        for feat in ALL_FEATURES:
            imp = agg_abs.get(feat, 0)
            pos = agg_pos.get(feat, 0)
            neg = abs(agg_neg.get(feat, 0))
            direction = 1 if pos > neg else (-1 if neg > pos else 0)
            rows.append({"feature": feat, "importance": imp, "direction": direction})
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame({"feature": ALL_FEATURES, "importance": [0.0] * len(ALL_FEATURES), "direction": [0] * len(ALL_FEATURES)})

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance"] = df["importance"].round(6)
    return df


def get_permutation_importance(pipeline, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """
    Compute permutation importance on held-out data.
    Returns DataFrame [feature, importance_mean, importance_std].
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
    )
    df = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_mean": result.importances_mean.round(6),
        "importance_std": result.importances_std.round(6),
    })
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def get_local_explanation(pipeline, form_data: dict) -> pd.DataFrame:
    """
    Approximate local feature importance for a single prediction via
    coefficient magnitudes (linear model) or tree importances weighted by
    feature value deviation from training median (tree models).

    Returns DataFrame [feature, contribution, direction].
    direction: +1 pushes toward Approved, -1 toward Rejected, 0 neutral.
    """
    from src.prediction import build_input_dataframe

    df_input = build_input_dataframe(form_data)
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    encoded_names = _get_feature_names_from_pipeline(pipeline)
    X_transformed = preprocessor.transform(df_input)

    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        contributions = coefs * X_transformed[0]
        agg_pos = _aggregate_to_original(np.maximum(contributions, 0), encoded_names)
        agg_neg = _aggregate_to_original(np.minimum(contributions, 0), encoded_names)
        agg_abs = _aggregate_to_original(np.abs(contributions), encoded_names)
        rows = []
        for feat in ALL_FEATURES:
            imp = agg_abs.get(feat, 0)
            pos = agg_pos.get(feat, 0)
            neg = abs(agg_neg.get(feat, 0))
            direction = 1 if pos > neg else (-1 if neg > pos else 0)
            rows.append({"feature": feat, "contribution": round(imp, 6), "direction": direction})
        return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)

    elif hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
        # Weight by absolute transformed value (deviation from zero after scaling)
        weighted = raw * np.abs(X_transformed[0])
        agg = _aggregate_to_original(weighted, encoded_names)
        df = pd.DataFrame(agg.items(), columns=["feature", "contribution"])
        df["direction"] = 0
        return df.sort_values("contribution", ascending=False).reset_index(drop=True)

    return pd.DataFrame({"feature": ALL_FEATURES, "contribution": [0.0]*len(ALL_FEATURES), "direction": [0]*len(ALL_FEATURES)})


def plot_global_importance(imp_df: pd.DataFrame, top_n: int = 15):
    """Return a horizontal bar chart of global feature importances."""
    import plotly.graph_objects as go

    df = imp_df.head(top_n).copy()
    df = df.sort_values("importance")  # ascending for horizontal bar
    colors = []
    for d in df["direction"]:
        if d == 1:
            colors.append("#10b981")
        elif d == -1:
            colors.append("#ef4444")
        else:
            colors.append("#3b82d4")

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Global Feature Importance (Top {top_n})",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(350, top_n * 28),
        margin=dict(l=150, r=20, t=60, b=40),
    )
    return fig


def plot_local_explanation(local_df: pd.DataFrame, prediction_label: str, top_n: int = 10):
    """Return a horizontal waterfall-style bar chart for a local prediction."""
    import plotly.graph_objects as go

    df = local_df.head(top_n).copy()
    df = df.sort_values("contribution")
    color_map = {1: "#10b981", -1: "#ef4444", 0: "#3b82d4"}
    colors = [color_map.get(int(d), "#3b82d4") for d in df.get("direction", [0]*len(df))]

    cont_col = "contribution" if "contribution" in df.columns else "importance_mean"

    fig = go.Figure(go.Bar(
        x=df[cont_col],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Local Explanation — Prediction: {prediction_label}",
        xaxis_title="Contribution to Prediction",
        height=max(300, top_n * 28),
        margin=dict(l=150, r=20, t=60, b=40),
    )
    return fig
