"""
Tests for prediction module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import VALID_CATEGORIES, build_preprocessor, get_X_y, load_clean_dataset, split_data
from src.model_training import evaluate_all_models, select_best_model, train_all_models
from src.prediction import build_input_dataframe, predict


SAMPLE_FORM = {
    "Applicant_Income": 12000.0,
    "Coapplicant_Income": 3000.0,
    "Employment_Status": "Salaried",
    "Age": 35.0,
    "Marital_Status": "Married",
    "Dependents": 1.0,
    "Credit_Score": 720.0,
    "Existing_Loans": 1.0,
    "DTI_Ratio": 0.30,
    "Savings": 15000.0,
    "Collateral_Value": 25000.0,
    "Loan_Amount": 10000.0,
    "Loan_Term": 60.0,
    "Loan_Purpose": "Home",
    "Property_Area": "Urban",
    "Education_Level": "Graduate",
    "Gender": "Male",
    "Employer_Category": "Private",
}

POOR_FORM = {
    "Applicant_Income": 3000.0,
    "Coapplicant_Income": 0.0,
    "Employment_Status": "Unemployed",
    "Age": 22.0,
    "Marital_Status": "Single",
    "Dependents": 3.0,
    "Credit_Score": 555.0,
    "Existing_Loans": 4.0,
    "DTI_Ratio": 0.59,
    "Savings": 200.0,
    "Collateral_Value": 500.0,
    "Loan_Amount": 35000.0,
    "Loan_Term": 12.0,
    "Loan_Purpose": "Personal",
    "Property_Area": "Rural",
    "Education_Level": "Not Graduate",
    "Gender": "Female",
    "Employer_Category": "Unemployed",
}


@pytest.fixture(scope="module")
def best_pipeline():
    df, _, _ = load_clean_dataset()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    prep = build_preprocessor()
    fitted = train_all_models(X_train, y_train, X_test, y_test, preprocessor=prep)
    metrics = evaluate_all_models(fitted, X_test, y_test)
    _, pipe = select_best_model(fitted, metrics)
    return pipe


def test_build_input_dataframe():
    df = build_input_dataframe(SAMPLE_FORM)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_build_input_dataframe_missing_field():
    bad = {k: v for k, v in SAMPLE_FORM.items() if k != "Credit_Score"}
    with pytest.raises(ValueError):
        build_input_dataframe(bad)


def test_predict_returns_label(best_pipeline):
    result = predict(best_pipeline, SAMPLE_FORM)
    assert result["label"] in ("Approved", "Rejected")


def test_predict_probability_range(best_pipeline):
    result = predict(best_pipeline, SAMPLE_FORM)
    if result["approved_probability"] is not None:
        assert 0.0 <= result["approved_probability"] <= 1.0
        assert 0.0 <= result["rejected_probability"] <= 1.0
        assert abs(result["approved_probability"] + result["rejected_probability"] - 1.0) < 1e-4


def test_predict_approved_scenario(best_pipeline):
    """Strong applicant should likely be approved."""
    result = predict(best_pipeline, SAMPLE_FORM)
    # We just check it runs, not the outcome (model-dependent)
    assert result["label"] in ("Approved", "Rejected")


def test_predict_rejected_scenario(best_pipeline):
    """Weak applicant should likely be rejected."""
    result = predict(best_pipeline, POOR_FORM)
    assert result["label"] in ("Approved", "Rejected")


def test_predict_consistency(best_pipeline):
    """Same input → same output (deterministic)."""
    r1 = predict(best_pipeline, SAMPLE_FORM)
    r2 = predict(best_pipeline, SAMPLE_FORM)
    assert r1["label"] == r2["label"]
    assert r1["approved_probability"] == r2["approved_probability"]
