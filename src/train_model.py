"""
Train and compare forecasting models on the processed electricity data.

Models:
  1. Naive baseline    -> predict lag_24h (yesterday, same hour)
  2. Linear Regression  -> simple linear model on features
  3. XGBoost Regressor  -> gradient boosted trees (main model)

Split: time-based (last 14 days = test set). No shuffling — this is a
forecasting problem, so we never train on the future.

Run: python src/train_model.py
Produces:
  data/predictions.csv   (test-set actual vs predicted, for the dashboard)
  data/metrics.csv        (MAE / RMSE / MAPE per model)
  data/model.json          (saved XGBoost model)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_PATH = DATA_DIR / "processed.csv"

FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month",
    "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_std_24h",
]
TARGET = "load_kw"

TEST_DAYS = 14


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_data():
    df = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
    return df


def time_split(df, test_days=TEST_DAYS):
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    train = df[df.index <= cutoff]
    test = df[df.index > cutoff]
    return train, test


def evaluate(name, y_true, y_pred, results):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    results.append({"model": name, "MAE": round(mae, 3),
                     "RMSE": round(rmse, 3), "MAPE_%": round(mp, 2)})
    print(f"{name:20s}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  MAPE={mp:6.2f}%")


def run():
    df = load_data()
    train, test = time_split(df)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows "
          f"(last {TEST_DAYS} days: {test.index.min()} -> {test.index.max()})")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = []
    preds = pd.DataFrame(index=test.index)
    preds["actual"] = y_test

    # 1. Naive baseline: predict yesterday's same hour
    preds["naive"] = X_test["lag_24h"]
    evaluate("Naive (lag-24h)", y_test, preds["naive"], results)

    # 2. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds["linear_reg"] = lr.predict(X_test)
    evaluate("Linear Regression", y_test, preds["linear_reg"], results)

    # 3. XGBoost
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    model.fit(X_train, y_train)
    preds["xgboost"] = model.predict(X_test)
    evaluate("XGBoost", y_test, preds["xgboost"], results)

    # Feature importance (nice to show the guide)
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)
    print("\nTop features (XGBoost):")
    print(importance.head(6).to_string())

    # Save everything the dashboard needs
    preds.to_csv(DATA_DIR / "predictions.csv")
    pd.DataFrame(results).to_csv(DATA_DIR / "metrics.csv", index=False)
    importance.to_csv(DATA_DIR / "feature_importance.csv", header=["importance"])
    model.save_model(str(DATA_DIR / "model.json"))

    print(f"\nSaved: predictions.csv, metrics.csv, feature_importance.csv, model.json")
    return preds, results


if __name__ == "__main__":
    run()
