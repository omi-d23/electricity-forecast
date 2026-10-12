"""
Data pipeline for the Electricity Consumption Forecasting project.

Source data: UCI "ElectricityLoadDiagrams20112014" dataset (321 clients,
15-min electricity load in kW, 2011-2014). We work with client "MT_001"
(renamed to `load_kw`) as our primary series, resampled to hourly.

Run: python src/pipeline.py
Produces: data/processed.csv  (clean, hourly, feature-engineered)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "electricity_raw.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed.csv"

# Which client column to use as our main series (there are 321: "1".."321")
CLIENT_COL = "1"


def load_raw(client_col: str = CLIENT_COL) -> pd.DataFrame:
    """Load only the date column + one client column (keeps memory sane)."""
    df = pd.read_csv(RAW_PATH, usecols=["date", client_col])
    df.columns = ["timestamp", "load_kw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # Resample raw 15-min data to hourly (mean load per hour)
    df = df.resample("1h").mean()

    # Handle missing values: interpolate short gaps, forward-fill the rest
    df["load_kw"] = df["load_kw"].interpolate(limit=6, method="linear")
    df["load_kw"] = df["load_kw"].ffill().bfill()

    # Drop the long leading stretch of zeros (client not yet metered)
    nonzero_start = df[df["load_kw"] > 0].index.min()
    df = df[df.index >= nonzero_start]

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek       # 0=Mon
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df.index.month
    df["day"] = df.index.day

    # Cyclical encodings (so the model knows hour 23 is close to hour 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Lag features (past values as predictors)
    df["lag_1h"] = df["load_kw"].shift(1)
    df["lag_24h"] = df["load_kw"].shift(24)   # same hour, yesterday
    df["lag_168h"] = df["load_kw"].shift(168)  # same hour, last week

    # Rolling stats
    df["roll_mean_24h"] = df["load_kw"].shift(1).rolling(24).mean()
    df["roll_std_24h"] = df["load_kw"].shift(1).rolling(24).std()

    df = df.dropna()
    return df


def run():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw()
    print(f"Raw rows: {len(df):,}")

    df = clean(df)
    print(f"After cleaning + hourly resample: {len(df):,} rows "
          f"({df.index.min()} -> {df.index.max()})")

    df = add_features(df)
    print(f"After feature engineering: {len(df):,} rows, {df.shape[1]} columns")

    df.to_csv(OUT_PATH)
    print(f"Saved processed data -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
