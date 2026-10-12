# Week 4: Logistics Predictive Modeling and Optimization

## Project Overview
This project predicts actual shipping duration in days using regression models trained on the DataCo logistics dataset. It compares a baseline with Linear Regression, Decision Tree, and Random Forest models, tunes a Decision Tree with time-series cross-validation, evaluates held-out predictions, and proposes operational optimization opportunities.

## Problem Definition
- **Target:** `Days for shipping (real)`
- **Dataset:** `data/DataCo_Logistics_Cleaned.csv`
- **Dataset size:** 180,519 records and 45 columns
- **Split:** Chronological training period followed by a later 20% holdout test set
- **Training efficiency:** An evenly spaced sample of up to 60,000 rows from the earlier training period is used for model fitting; all 36,104 later holdout rows are used for final evaluation.

## Leakage Prevention
The model excludes post-outcome fields such as `shipping date (DateOrders)`, `Delivery Status`, `Late_delivery_risk`, `Delivery Delay`, and `Is Delayed`. Scheduled shipping duration is retained because it is generally available at order time. Confirm feature availability before any real deployment.

## Models and Metrics
Models compared: Median Baseline, Linear Regression, Decision Tree, Random Forest, and a tuned Decision Tree. Evaluation metrics are:
- **MAE:** Average absolute prediction error in days.
- **RMSE:** Error metric that penalizes larger mistakes more strongly.
- **R²:** Variation explained relative to a mean-prediction baseline.

The script chooses the best model by measured holdout RMSE. The report includes the measured results and cautions that retrospective metrics do not guarantee future performance.

## Project Structure
```text
Week4_Logistics_Predictive_Optimization/
├── data/
│   └── DataCo_Logistics_Cleaned.csv
├── src/
│   └── train_model.py
├── outputs/
│   ├── figures/
│   └── test_predictions.csv
├── reports/
│   ├── model_comparison.csv
│   ├── model_summary.json
│   ├── top_feature_importance.csv
│   └── shipping_mode_test_summary.csv
├── models/
│   └── best_shipping_duration_model.joblib
├── requirements.txt
├── README.md
├── create_report.py
└── Week4_Logistics_Predictive_Optimization_Report.docx
```

## Run Locally
Use Python 3.10 or newer. From the project root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
```

On macOS/Linux, activate the environment with `source venv/bin/activate` instead.

## Key Outputs
- `reports/model_comparison.csv` — MAE, RMSE, and R² for each model.
- `reports/model_summary.json` — split details, selected model, and tuning results.
- `reports/top_feature_importance.csv` — feature importance for the selected tree-based model.
- `outputs/test_predictions.csv` — actual and predicted values with absolute errors.
- `outputs/figures/` — model comparison, actual-versus-predicted, residual, and feature-importance charts.
- `models/best_shipping_duration_model.joblib` — fitted preprocessing/model pipeline.

## Optimization Opportunities
Use predictions to prioritize orders for review, plan workload, and monitor errors by shipping mode and region. Do not automatically reroute shipments based only on this model. True route or cost optimization requires additional data such as road-network distances, traffic, carrier prices, vehicle capacity, and delivery constraints.

## Limitations
The model's performance is retrospective and may change over time. The available dataset does not contain all operational inputs required for causal conclusions, live route planning, or verified cost-savings calculations. Recommendations should be tested through a controlled pilot and monitored before production use.

## Author
**Sparsh Kulshrestha**  
MCA | Python | Data Analysis | Predictive Modeling
