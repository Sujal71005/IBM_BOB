# LoanVision AI

An educational AI-powered loan analytics and prediction platform built with Python, Streamlit, Pandas, Plotly, and Scikit-learn.

> ⚠️ **Disclaimer:** This is an educational prototype using a synthetic dataset. It must not be used to make actual lending or credit decisions.

---

## Features

- **Executive Overview** — KPI cards, approval/rejection charts, loan purpose and employment distribution
- **Exploratory Data Analysis** — Interactive distributions, approval rates by category, scatter plots, correlation heatmap, box plots with filter controls
- **Machine Learning Model Lab** — Train/compare Logistic Regression, Random Forest, and Gradient Boosting; confusion matrices, metrics table, ROC curves
- **Loan Approval Predictor** — Real-time prediction form with probability gauge and local explanation
- **Model Explainability** — Global feature importance, permutation importance on held-out data, local explanation per prediction
- **Data Explorer** — Searchable/filterable table with pagination and CSV export
- **Project Information** — Full documentation of methodology, features, limitations, and ethics

---

## Technology Stack

| Layer | Library |
|-------|---------|
| Dashboard | Streamlit 1.32+ |
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualisation | Plotly |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Testing | Pytest |

---

## Dataset Schema

File: `data/loan_approval_data.csv` (1,000 rows)

| Column | Type | Description |
|--------|------|-------------|
| Applicant_ID | ID | Excluded from modelling |
| Applicant_Income | Numerical | Primary applicant's income |
| Coapplicant_Income | Numerical | Co-applicant's income |
| Employment_Status | Categorical | Contract / Salaried / Self-employed / Unemployed |
| Age | Numerical | Applicant age |
| Marital_Status | Categorical | Married / Single |
| Dependents | Numerical | Financial dependents (0–3) |
| Credit_Score | Numerical | 550–800 |
| Existing_Loans | Numerical | Number of current active loans |
| DTI_Ratio | Numerical | Debt-to-income ratio (0.10–0.60) |
| Savings | Numerical | Liquid savings |
| Collateral_Value | Numerical | Pledged collateral value |
| Loan_Amount | Numerical | Requested loan amount |
| Loan_Term | Numerical | Duration in months (12–84) |
| Loan_Purpose | Categorical | Business / Car / Education / Home / Personal |
| Property_Area | Categorical | Rural / Semiurban / Urban |
| Education_Level | Categorical | Graduate / Not Graduate |
| Gender | Categorical | Female / Male |
| Employer_Category | Categorical | Business / Government / MNC / Private / Unemployed |
| **Loan_Approved** | **Target** | **Yes / No (→ 1 / 0)** |

---

## Installation

```bash
# 1. Clone or download the project
cd loanvision-ai

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
# From the loanvision-ai/ directory:
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

On first launch, navigate to **Model Lab** and click **Train Models** to train and save the pipeline.

---

## Model Training (standalone)

```bash
python _run_training.py
```

This trains all three classifiers, evaluates on the held-out test set, selects the best by ROC-AUC, and saves the pipeline to `models/loanvision_pipeline.joblib` and metrics to `reports/model_metrics.json`.

---

## Machine Learning Workflow

1. **Load & Validate** — Dataset loaded, 50 rows with missing `Loan_Approved` dropped.
2. **Clean** — Duplicate removal, type conversion, empty-string → NaN normalisation.
3. **Split** — 80/20 stratified train/test split (random_state=42).
4. **Preprocess (inside Pipeline):**
   - Numerical: median imputation → StandardScaler
   - Categorical: mode imputation → OneHotEncoder (unknown categories ignored)
   - All fit on training data only — no leakage.
5. **Train** — Three classifiers trained with class balancing where supported.
6. **Evaluate** — Accuracy, Precision, Recall, F1, ROC-AUC on held-out test set.
7. **Select** — Best model chosen by ROC-AUC (appropriate for imbalanced classes).
8. **Save** — Full pipeline (preprocessor + classifier) saved as a single Joblib artifact.

---

## Evaluation Metrics

| Metric | Why It Matters |
|--------|----------------|
| Accuracy | Overall correctness — misleading when classes are imbalanced |
| Precision | Of predicted approvals, how many were actually approved |
| Recall | Of actual approvals, how many did the model catch |
| F1-Score | Harmonic mean of precision and recall |
| **ROC-AUC** | **Primary criterion** — measures discrimination across all thresholds; robust to class imbalance |

### Actual Results (held-out test set, 20% of 950 records)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Gradient Boosting | 0.9579 | 0.8939 | 0.9833 | 0.9365 | 0.9873 |
| Random Forest | 0.9211 | 0.9091 | 0.8333 | 0.8696 | 0.9834 |
| Logistic Regression | 0.8474 | 0.7183 | 0.8500 | 0.7786 | 0.9197 |

**Selected: Gradient Boosting** (highest ROC-AUC = 0.9873)

---

## Project Structure

```
loanvision-ai/
├── app.py                      # Main Streamlit application (7 pages)
├── requirements.txt
├── README.md
├── _run_training.py            # Standalone training script
├── data/
│   └── loan_approval_data.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Load, validate, clean, preprocess
│   ├── model_training.py       # Train, evaluate, select, save/load
│   ├── evaluation.py           # Plotting helpers for metrics
│   ├── prediction.py           # Predict from form data
│   └── explainability.py       # Global & local feature importance
├── models/
│   └── loanvision_pipeline.joblib   # Saved after training
├── reports/
│   └── model_metrics.json           # Saved after training
└── tests/
    ├── test_data_processing.py   # 17 tests
    ├── test_model_training.py    # 5 tests
    └── test_prediction.py        # 7 tests
```

---

## Running Tests

```bash
# From the loanvision-ai/ directory:
pytest tests/ -v
```

Expected: **29 tests pass**.

---

## Known Limitations

- The dataset is synthetic/educational. Real-world model performance may differ significantly.
- No hyperparameter optimisation was applied. Performance could be improved with grid search or Bayesian optimisation.
- The test set (~190 rows) is small; metric confidence intervals are wide.
- Probability outputs are not calibrated (Platt scaling / isotonic regression not applied).
- Local explanations are approximations, not exact SHAP values.
- No formal fairness audit across demographic groups has been performed.
- The application is designed for single-user local use, not multi-user production deployment.
