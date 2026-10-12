"""
Forward Forecasting — predicts the NEXT N hours beyond the end of the
dataset, using the trained XGBoost model.

This is different from train_model.py, which only evaluates the model
on known past data (backtesting). This script does genuine forecasting:
it has no actual future values to check against, so it must predict one
hour at a time and feed each prediction back in as a lag feature for the
next hour (this is called "recursive forecasting").

Run: python src/forecast.py --hours 24
Produces: data/forecast.csv  (timestamp, predicted_load_kw)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_PATH = DATA_DIR / "processed.csv"
MODEL_PATH = DATA_DIR / "model.json"
OUT_PATH = DATA_DIR / "forecast.csv"

FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month",
    "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_std_24h",
]


def build_features_for_timestamp(ts, history: pd.Series):
    """
    Build the feature row for a single future timestamp, using
    `history` — a Series of load values (actual + predicted-so-far)
    indexed by timestamp, going back far enough to cover a 168-hour lag.
    """
    hour = ts.hour
    dow = ts.dayofweek

    def lag_value(hours_back):
        t = ts - pd.Timedelta(hours=hours_back)
        return history.get(t, np.nan)

    lag_1h = lag_value(1)
    lag_24h = lag_value(24)
    lag_168h = lag_value(168)

    # rolling mean/std of the 24 hours before this timestamp
    window_start = ts - pd.Timedelta(hours=24)
    window = history[(history.index >= window_start) & (history.index < ts)]
    roll_mean_24h = window.mean() if len(window) > 0 else lag_24h
    roll_std_24h = window.std() if len(window) > 1 else 0.0

    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": int(dow >= 5),
        "month": ts.month,
        "lag_1h": lag_1h,
        "lag_24h": lag_24h,
        "lag_168h": lag_168h,
        "roll_mean_24h": roll_mean_24h,
        "roll_std_24h": roll_std_24h,
    }


def run(hours_ahead: int = 24):
    df = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
    history = df["load_kw"].copy()  # actual load, indexed by timestamp

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    last_ts = history.index.max()
    print(f"Last known reading: {last_ts}")
    print(f"Forecasting next {hours_ahead} hours...")

    forecasts = []
    for step in range(1, hours_ahead + 1):
        ts = last_ts + pd.Timedelta(hours=step)
        feat = build_features_for_timestamp(ts, history)
        X = pd.DataFrame([feat])[FEATURES]

        pred = float(model.predict(X)[0])
        history[ts] = pred  # feed prediction back in for future lags
        forecasts.append({"timestamp": ts, "predicted_load_kw": round(pred, 3)})

    result = pd.DataFrame(forecasts).set_index("timestamp")
    result.to_csv(OUT_PATH)

    print(f"\nForecast (next {hours_ahead}h):")
    print(result.to_string())
    print(f"\nSaved -> {OUT_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24,
                         help="Number of hours ahead to forecast (default: 24)")
    args = parser.parse_args()
    run(args.hours)
