"""
Tests for data_processing module.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
    build_preprocessor,
    clean_dataset,
    get_X_y,
    load_raw_dataset,
    split_data,
    validate_dataset,
    load_clean_dataset,
)


@pytest.fixture
def raw_df():
    return load_raw_dataset()


@pytest.fixture
def clean_df(raw_df):
    df, _ = clean_dataset(raw_df)
    return df


# ── Loading ─────────────────────────────────────────────────────────────────────

def test_load_raw_dataset_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) > 0


def test_load_raw_dataset_has_expected_columns(raw_df):
    for col in ALL_FEATURES + [TARGET_COL, "Applicant_ID"]:
        assert col in raw_df.columns, f"Expected column '{col}' not found"


def test_missing_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_dataset("/nonexistent/path/file.csv")


# ── Validation ───────────────────────────────────────────────────────────────────

def test_validate_returns_dict(raw_df):
    report = validate_dataset(raw_df)
    assert isinstance(report, dict)
    for key in ("total_rows", "missing_target_rows", "duplicate_rows", "warnings"):
        assert key in report


def test_validate_detects_missing_target(raw_df):
    report = validate_dataset(raw_df)
    assert report["missing_target_rows"] == 50


# ── Cleaning ─────────────────────────────────────────────────────────────────────

def test_clean_drops_missing_target_rows(raw_df):
    df, log = clean_dataset(raw_df)
    assert log["dropped_missing_target"] == 50
    assert df[TARGET_COL].isna().sum() == 0


def test_clean_target_is_binary(clean_df):
    assert set(clean_df[TARGET_COL].unique()).issubset({0, 1})


def test_clean_no_empty_strings(clean_df):
    for col in ALL_FEATURES:
        if col in clean_df.columns:
            str_mask = clean_df[col].apply(lambda x: x == "")
            assert not str_mask.any(), f"Empty string found in {col}"


def test_clean_final_count(clean_df):
    # Should have 950 rows (1000 - 50 missing target)
    assert len(clean_df) >= 940  # allow for possible duplicates


# ── Feature / target split ────────────────────────────────────────────────────────

def test_get_X_y_shapes(clean_df):
    X, y = get_X_y(clean_df)
    assert len(X) == len(y)
    assert list(X.columns) == ALL_FEATURES


def test_get_X_y_no_target_in_X(clean_df):
    X, y = get_X_y(clean_df)
    assert TARGET_COL not in X.columns
    assert "Applicant_ID" not in X.columns


def test_get_X_y_target_binary(clean_df):
    _, y = get_X_y(clean_df)
    assert set(y.unique()).issubset({0, 1})


# ── Split ─────────────────────────────────────────────────────────────────────────

def test_split_data_sizes(clean_df):
    X, y = get_X_y(clean_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    total = len(X)
    assert abs(len(X_test) - int(total * 0.2)) <= 2


def test_split_data_no_overlap(clean_df):
    X, y = get_X_y(clean_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert len(train_idx & test_idx) == 0


# ── Preprocessor ─────────────────────────────────────────────────────────────────

def test_preprocessor_transforms(clean_df):
    X, y = get_X_y(clean_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    prep = build_preprocessor()
    prep.fit(X_train)
    transformed = prep.transform(X_test)
    assert transformed.shape[0] == len(X_test)
    assert transformed.shape[1] > len(ALL_FEATURES)  # OHE expands categoricals


def test_preprocessor_no_nans_after_transform(clean_df):
    X, y = get_X_y(clean_df)
    X_train, X_test, _, _ = split_data(X, y)
    prep = build_preprocessor()
    prep.fit(X_train)
    transformed = prep.transform(X_test)
    assert not np.isnan(transformed).any()


# ── Integration ───────────────────────────────────────────────────────────────────

def test_load_clean_dataset():
    df, val_report, clean_log = load_clean_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert clean_log["dropped_missing_target"] == 50
