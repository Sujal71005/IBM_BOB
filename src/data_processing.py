"""
data_processing.py
------------------
Dataset loading, validation, cleaning, and preprocessing for LoanVision AI.

Preprocessing decisions:
  - Rows with missing Loan_Approved are dropped (target unknown — cannot train/evaluate on them).
  - Applicant_ID is excluded from features (identifier, not predictive).
  - Numerical missing values are imputed with the median (robust to skew/outliers).
  - Categorical missing values are imputed with the most-frequent value.
  - All imputation statistics are derived from training data only to prevent leakage.
  - One-hot encoding is applied to all categorical features.
  - Numerical features are standard-scaled inside the sklearn Pipeline.
  - Duplicate rows (all columns identical) are dropped.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Column metadata ───────────────────────────────────────────────────────────

TARGET_COL = "Loan_Approved"
ID_COL = "Applicant_ID"

NUMERICAL_FEATURES = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Dependents",
]

CATEGORICAL_FEATURES = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

# Union: all model input features (ordered, stable)
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Valid category values discovered from the full dataset
VALID_CATEGORIES: dict[str, list[str]] = {
    "Employment_Status": ["Contract", "Salaried", "Self-employed", "Unemployed"],
    "Marital_Status": ["Married", "Single"],
    "Loan_Purpose": ["Business", "Car", "Education", "Home", "Personal"],
    "Property_Area": ["Rural", "Semiurban", "Urban"],
    "Education_Level": ["Graduate", "Not Graduate"],
    "Gender": ["Female", "Male"],
    "Employer_Category": ["Business", "Government", "MNC", "Private", "Unemployed"],
}

DATA_PATH = Path(__file__).parent.parent / "data" / "loan_approval_data.csv"


# ─── Loading ────────────────────────────────────────────────────────────────────

def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV; raise FileNotFoundError with a helpful message if absent."""
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{p}'. "
            "Place loan_approval_data.csv inside the data/ folder."
        )
    df = pd.read_csv(p)
    return df


# ─── Validation ─────────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the loaded DataFrame and return a report dict.
    Does NOT raise — callers inspect the report and decide.
    """
    report: dict = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_target_rows": int(df[TARGET_COL].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
        "out_of_range": {},
        "unexpected_categories": {},
        "warnings": [],
    }

    # Numerical range checks (soft — we warn, not drop)
    range_checks = {
        "Age": (18, 100),
        "Credit_Score": (300, 850),
        "DTI_Ratio": (0.0, 1.0),
        "Dependents": (0, 10),
        "Existing_Loans": (0, 20),
        "Applicant_Income": (0, None),
        "Loan_Amount": (0, None),
    }
    for col, (lo, hi) in range_checks.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        mask = pd.Series(False, index=series.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        if mask.any():
            report["out_of_range"][col] = int(mask.sum())
            report["warnings"].append(
                f"{col}: {mask.sum()} value(s) outside expected range [{lo}, {hi}]."
            )

    # Categorical value checks
    for col, valid in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        unexpected = df[col].dropna()[~df[col].dropna().isin(valid)].unique().tolist()
        if unexpected:
            report["unexpected_categories"][col] = unexpected
            report["warnings"].append(
                f"{col}: unexpected value(s) {unexpected}."
            )

    return report


# ─── Cleaning ────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw DataFrame.

    Returns (cleaned_df, cleaning_log).

    Steps:
      1. Drop rows where Loan_Approved is missing (cannot use as training/eval rows).
      2. Drop exact duplicate rows.
      3. Convert column types.
      4. Strip whitespace from string columns.
    """
    log: dict = {}
    df = df.copy()

    # 1. Drop rows missing target
    before = len(df)
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL].astype(str).str.strip() != "")]
    log["dropped_missing_target"] = before - len(df)

    # 2. Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    log["dropped_duplicates"] = before - len(df)

    # 3. Strip whitespace in all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 4. Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)

    # 5. Convert numerics
    for col in NUMERICAL_FEATURES + [ID_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Encode target: Yes → 1, No → 0
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    log["dropped_invalid_target"] = int(df[TARGET_COL].isna().sum())
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    log["final_row_count"] = len(df)
    log["missing_after_clean"] = df[ALL_FEATURES].isna().sum().to_dict()

    return df, log


# ─── Feature / target split ──────────────────────────────────────────────────────

def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X uses only ALL_FEATURES and y is the binary target."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ─── Sklearn preprocessor (used inside Pipeline) ─────────────────────────────────

def build_preprocessor():
    """
    Return a ColumnTransformer that:
      - Imputes + scales numerical features
      - Imputes + one-hot-encodes categorical features
    Must be used inside an sklearn Pipeline; fit only on training data.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numerical_pipeline = SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = SKPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return preprocessor


# ─── Train / test split ───────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# ─── Convenience: load + clean in one call ────────────────────────────────────────

def load_clean_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load, validate, and clean the dataset.
    Returns (clean_df, validation_report, cleaning_log).
    """
    raw = load_raw_dataset(path)
    validation_report = validate_dataset(raw)
    clean_df, cleaning_log = clean_dataset(raw)
    return clean_df, validation_report, cleaning_log
