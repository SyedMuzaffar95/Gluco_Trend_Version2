import os
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from models.load import load_dataset
from models.preprocessing import prepare_features


MODEL_DIR = "models/saved"
MODEL_PATH = os.path.join(MODEL_DIR, "glucose_ridge_model.joblib")
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "plots"
)
os.makedirs(PLOT_DIR, exist_ok=True)


def chronological_split(X, y, df, test_size=0.20):
    """Split by time to avoid using future observations for training."""
    timestamps = pd.to_datetime(
        df["timestamp"], dayfirst=True, errors="coerce"
    )
    order = np.argsort(timestamps.values)
    cut = int(len(order) * (1 - test_size))

    train_idx = order[:cut]
    test_idx = order[cut:]

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def _compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MSE": round(mse, 4),
        "RMSE": round(mse ** 0.5, 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def _generate_regression_plots(y_test, ridge_preds, lr_preds, ridge_metrics, lr_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar chart: MAE & MSE
    metric_names = ["MAE", "MSE"]
    ridge_vals = [ridge_metrics[m] for m in metric_names]
    lr_vals = [lr_metrics[m] for m in metric_names]
    x = np.arange(len(metric_names))
    width = 0.35
    axes[0, 0].bar(x - width / 2, ridge_vals, width, label="Ridge", color="#2ecc71")
    axes[0, 0].bar(x + width / 2, lr_vals, width, label="Linear Regression", color="#3498db")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_names)
    axes[0, 0].set_title("MAE & MSE Comparison")
    axes[0, 0].legend()

    # Bar chart: RMSE & R2
    metric_names2 = ["RMSE", "R2"]
    ridge_vals2 = [ridge_metrics[m] for m in metric_names2]
    lr_vals2 = [lr_metrics[m] for m in metric_names2]
    x2 = np.arange(len(metric_names2))
    axes[0, 1].bar(x2 - width / 2, ridge_vals2, width, label="Ridge", color="#2ecc71")
    axes[0, 1].bar(x2 + width / 2, lr_vals2, width, label="Linear Regression", color="#3498db")
    axes[0, 1].set_xticks(x2)
    axes[0, 1].set_xticklabels(metric_names2)
    axes[0, 1].set_title("RMSE & R² Comparison")
    axes[0, 1].legend()

    # Actual vs Predicted - Ridge
    axes[1, 0].scatter(y_test, ridge_preds, alpha=0.4, s=10, color="#2ecc71")
    min_val = min(y_test.min(), ridge_preds.min())
    max_val = max(y_test.max(), ridge_preds.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    axes[1, 0].set_xlabel("Actual Glucose")
    axes[1, 0].set_ylabel("Predicted Glucose")
    axes[1, 0].set_title("Ridge - Actual vs Predicted")

    # Actual vs Predicted - Linear Regression
    axes[1, 1].scatter(y_test, lr_preds, alpha=0.4, s=10, color="#3498db")
    min_val2 = min(y_test.min(), lr_preds.min())
    max_val2 = max(y_test.max(), lr_preds.max())
    axes[1, 1].plot([min_val2, max_val2], [min_val2, max_val2], "r--", linewidth=1.5)
    axes[1, 1].set_xlabel("Actual Glucose")
    axes[1, 1].set_ylabel("Predicted Glucose")
    axes[1, 1].set_title("Linear Regression - Actual vs Predicted")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "regression_metrics.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return "regression_metrics.png"


def train_regression():
    df = load_dataset()
    X, y, preprocessor, numeric_features, categorical_features = prepare_features(df)

    X_train, X_test, y_train, y_test = chronological_split(X, y, df)

    ridge_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    lr_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    ridge_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    ridge_preds = ridge_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)

    ridge_metrics = _compute_metrics(y_test, ridge_preds)
    lr_metrics = _compute_metrics(y_test, lr_preds)

    metrics = {
        "MAE": ridge_metrics["MAE"],
        "MSE": ridge_metrics["MSE"],
        "RMSE": ridge_metrics["RMSE"],
        "R2": ridge_metrics["R2"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_before_encoding": X.shape[1],
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "lr_MAE": lr_metrics["MAE"],
        "lr_MSE": lr_metrics["MSE"],
        "lr_RMSE": lr_metrics["RMSE"],
        "lr_R2": lr_metrics["R2"],
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ridge_model, MODEL_PATH)

    plot_image = _generate_regression_plots(
        y_test, ridge_preds, lr_preds, ridge_metrics, lr_metrics
    )

    results = pd.DataFrame({
        "Actual Glucose": y_test.values,
        "Ridge Predicted": ridge_preds.round(2),
        "LR Predicted": lr_preds.round(2),
        "Ridge Abs Error": np.abs(y_test.values - ridge_preds).round(2),
        "LR Abs Error": np.abs(y_test.values - lr_preds).round(2),
    }, index=y_test.index)

    return metrics, results, MODEL_PATH, plot_image


def predict(features: pd.DataFrame):
    if not os.path.exists(MODEL_PATH):
        train_regression()

    model = joblib.load(MODEL_PATH)
    return model.predict(features)
