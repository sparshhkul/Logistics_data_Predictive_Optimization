"""Week 4: Predictive modeling and optimization for logistics shipping duration.

Uses a chronological train/test split and excludes post-delivery fields to reduce
label leakage. Run from the project root:
    python src/train_model.py
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DataCo_Logistics_Cleaned.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for p in [OUT, FIG, REPORTS, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

TARGET = "Days for shipping (real)"
DATE_COL = "order date (DateOrders)"

# Load the full available dataset.
df = pd.read_csv(DATA_PATH, low_memory=False)
required = [TARGET, DATE_COL]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# Parse order date and retain rows with a valid target and timestamp.
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
initial_rows = len(df)
df = df.dropna(subset=[TARGET, DATE_COL]).copy()
df = df[df[TARGET] >= 0].copy()

# Feature engineering uses only order-time information. Delivery status, actual
# shipping date, delay flags, delivery risk, and the target itself are excluded.
df["Order Month"] = df[DATE_COL].dt.month
df["Order Day Of Week"] = df[DATE_COL].dt.dayofweek

def find_col(candidates):
    return next((c for c in candidates if c in df.columns), None)

numeric_candidates = [
    "Days for shipment (scheduled)", "Order Item Quantity",
    "Order Item Discount Rate", "Order Item Product Price",
    "Order Item Profit Ratio", "Latitude", "Longitude", "Order Month",
    "Order Day Of Week"
]
categorical_candidates = [
    "Type", "Category Name", "Customer Segment", "Market",
    "Order Region", "Shipping Mode", "Department Name"
]
numeric_features = [c for c in numeric_candidates if c in df.columns]
cat_features = [c for c in categorical_candidates if c in df.columns]
features = numeric_features + cat_features
if not features:
    raise ValueError("No eligible features were found in the dataset.")

# Sort chronologically before splitting to simulate predicting later orders.
df = df.sort_values(DATE_COL).reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
# To keep execution practical on a standard laptop, train on an evenly spaced
# chronological sample of up to 60,000 rows from the training period. The holdout
# test set remains the full later 20% of records.
model_train_n = min(60000, len(train_df))
model_train_idx = np.linspace(0, len(train_df) - 1, model_train_n, dtype=int)
model_train_df = train_df.iloc[model_train_idx].copy()
X_train, y_train = model_train_df[features], model_train_df[TARGET]
X_test, y_test = test_df[features], test_df[TARGET]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", cat_pipe, cat_features)
], remainder="drop")

models = {
    "Median Baseline": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=35, max_depth=12, min_samples_leaf=10,
                                             max_features=0.8, n_jobs=2, random_state=42)
}
results = []
predictions = {}
fitted = {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    predictions[name] = pred
    fitted[name] = pipe
    results.append({
        "Model": name,
        "MAE_days": float(mean_absolute_error(y_test, pred)),
        "RMSE_days": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred))
    })

results_df = pd.DataFrame(results).sort_values("RMSE_days").reset_index(drop=True)
results_df.to_csv(REPORTS / "model_comparison.csv", index=False)

# Hyperparameter tuning: a small, chronological cross-validation search for the
# decision tree. Keep the search bounded for reproducible execution.
# Search is performed on up to 40,000 chronologically ordered training rows.
search_n = min(15000, len(X_train))
X_search, y_search = X_train.iloc[:search_n], y_train.iloc[:search_n]
search_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])
cv = TimeSeriesSplit(n_splits=2)
search = GridSearchCV(
    search_pipe,
    param_grid={"model__max_depth": [8, 12], "model__min_samples_leaf": [10, 25]},
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=1,
    refit=True,
    error_score="raise"
)
search.fit(X_search, y_search)
tuned_pred = search.best_estimator_.predict(X_test)
tuned_metrics = {
    "Model": "Tuned Decision Tree",
    "MAE_days": float(mean_absolute_error(y_test, tuned_pred)),
    "RMSE_days": float(np.sqrt(mean_squared_error(y_test, tuned_pred))),
    "R2": float(r2_score(y_test, tuned_pred))
}
results_df = pd.concat([results_df, pd.DataFrame([tuned_metrics])], ignore_index=True)
results_df = results_df.sort_values("RMSE_days").reset_index(drop=True)
results_df.to_csv(REPORTS / "model_comparison.csv", index=False)

# Choose the best measured model from the held-out test metrics.
all_predictions = dict(predictions)
all_predictions["Tuned Decision Tree"] = tuned_pred
best_name = results_df.iloc[0]["Model"]
if best_name == "Tuned Decision Tree":
    best_pipe = search.best_estimator_
    best_pred = tuned_pred
else:
    best_pipe = fitted[best_name]
    best_pred = predictions[best_name]
dump(best_pipe, MODELS / "best_shipping_duration_model.joblib")

# Save test predictions for audit and later exploration.
pred_out = test_df[[DATE_COL, TARGET]].copy()
pred_out["Predicted Shipping Days"] = best_pred
pred_out["Absolute Error Days"] = np.abs(pred_out[TARGET] - pred_out["Predicted Shipping Days"])
pred_out.to_csv(OUT / "test_predictions.csv", index=False)

# Baseline and model evaluation summary.
summary = {
    "dataset_file": DATA_PATH.name,
    "rows_before_filtering": int(initial_rows),
    "rows_used": int(len(df)),
    "columns_in_source": int(df.shape[1] - 2),
    "target": TARGET,
    "target_mean_days": float(df[TARGET].mean()),
    "target_median_days": float(df[TARGET].median()),
    "train_rows": int(len(X_train)),
    "full_chronological_training_period_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "train_period_start": str(train_df[DATE_COL].min()),
    "train_period_end": str(train_df[DATE_COL].max()),
    "test_period_start": str(test_df[DATE_COL].min()),
    "test_period_end": str(test_df[DATE_COL].max()),
    "numeric_features": numeric_features,
    "categorical_features": cat_features,
    "excluded_post_outcome_fields": ["shipping date (DateOrders)", "Delivery Status", "Late_delivery_risk", "Delivery Delay", "Is Delayed"],
    "best_model_by_test_rmse": str(best_name),
    "best_model_metrics": {
        "MAE_days": float(mean_absolute_error(y_test, best_pred)),
        "RMSE_days": float(np.sqrt(mean_squared_error(y_test, best_pred))),
        "R2": float(r2_score(y_test, best_pred))
    },
    "tuned_decision_tree_best_params": search.best_params_,
    "tuned_decision_tree_cv_rmse_days": float(-search.best_score_),
    "note": "Chronological holdout evaluation. Metrics are retrospective on this dataset and do not guarantee future operational performance."
}
with open(REPORTS / "model_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Visualization 1: model comparison.
plt.figure(figsize=(10, 6))
plot_df = results_df.sort_values("RMSE_days", ascending=True)
plot_df.set_index("Model")["RMSE_days"].sort_values().plot(kind="barh")
plt.title("Model Comparison by Test RMSE")
plt.xlabel("RMSE (days; lower is better)")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(FIG / "01_model_comparison_rmse.png", dpi=160)
plt.close()

# Visualization 2: actual vs predicted for the selected model.
plt.figure(figsize=(7, 7))
plot_sample = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": best_pred}).sample(
    n=min(6000, len(y_test)), random_state=42)
plt.scatter(plot_sample["Actual"], plot_sample["Predicted"], alpha=0.35, s=18)
lo = min(plot_sample["Actual"].min(), plot_sample["Predicted"].min())
hi = max(plot_sample["Actual"].max(), plot_sample["Predicted"].max())
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.title(f"Actual vs Predicted Shipping Days — {best_name}")
plt.tight_layout()
plt.savefig(FIG / "02_actual_vs_predicted.png", dpi=160)
plt.close()

# Visualization 3: residual distribution.
residuals = y_test.to_numpy() - best_pred
plt.figure(figsize=(9, 5))
plt.hist(residuals, bins=35)
plt.axvline(0, linestyle="--")
plt.title("Prediction Residual Distribution")
plt.xlabel("Actual minus predicted shipping days")
plt.tight_layout()
plt.savefig(FIG / "03_residual_distribution.png", dpi=160)
plt.close()

# Visualization 4: feature importance when the selected model supports it.
model_component = best_pipe.named_steps["model"]
if hasattr(model_component, "feature_importances_"):
    try:
        feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
        importance = pd.DataFrame({"Feature": feature_names, "Importance": model_component.feature_importances_})
        importance = importance.sort_values("Importance", ascending=False).head(15)
        importance.to_csv(REPORTS / "top_feature_importance.csv", index=False)
        plt.figure(figsize=(10, 7))
        importance.set_index("Feature")["Importance"].sort_values().plot(kind="barh")
        plt.title(f"Top Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(FIG / "04_feature_importance.png", dpi=160)
        plt.close()
    except Exception as e:
        summary["feature_importance_plot_note"] = str(e)

# Operational analysis by shipping mode (descriptive, not causal).
if "Shipping Mode" in test_df.columns:
    mode = test_df[["Shipping Mode", TARGET]].copy()
    mode["Absolute Prediction Error"] = np.abs(y_test.to_numpy() - best_pred)
    mode_summary = mode.groupby("Shipping Mode", dropna=False).agg(
        records=(TARGET, "size"), mean_actual_days=(TARGET, "mean"),
        median_actual_days=(TARGET, "median"), mean_absolute_error_days=("Absolute Prediction Error", "mean")
    ).reset_index().sort_values("mean_actual_days", ascending=False)
    mode_summary.to_csv(REPORTS / "shipping_mode_test_summary.csv", index=False)

# Save updated summary after possible optional notes.
with open(REPORTS / "model_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Week 4 model training complete.")
print(f"Rows used: {len(df):,}; training-period rows: {len(train_df):,}; model-fit sample: {len(X_train):,}; test: {len(test_df):,}")
print("Model comparison (lower MAE/RMSE is better; higher R2 is better):")
print(results_df.to_string(index=False))
print(f"Best model: {best_name}")
print("Best model metrics:", summary["best_model_metrics"])
print("Tuned decision tree parameters:", search.best_params_)
print(f"Outputs saved under: {ROOT}")
