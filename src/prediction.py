"""
prediction.py
-------------
Make predictions using the saved LoanVision AI pipeline.

The prediction form must use exactly the same feature names and ordering
as ALL_FEATURES in data_processing.py.  This module enforces that contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form dict to a single-row DataFrame with correct column names/types.
    Raises ValueError on missing required fields.
    """
    missing = [f for f in ALL_FEATURES if f not in form_data]
    if missing:
        raise ValueError(f"Missing input field(s): {missing}")

    row = {col: [form_data[col]] for col in ALL_FEATURES}
    df = pd.DataFrame(row)

    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def predict(pipeline, form_data: dict) -> dict:
    """
    Run end-to-end prediction.

    Returns:
        {
            "label": "Approved" | "Rejected",
            "approved_probability": float | None,
            "rejected_probability": float | None,
        }
    """
    df = build_input_dataframe(form_data)
    pred = pipeline.predict(df)[0]
    label = "Approved" if pred == 1 else "Rejected"

    approved_prob = None
    rejected_prob = None
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba(df)[0]
        rejected_prob = round(float(proba[0]), 4)
        approved_prob = round(float(proba[1]), 4)

    return {
        "label": label,
        "approved_probability": approved_prob,
        "rejected_probability": rejected_prob,
    }
