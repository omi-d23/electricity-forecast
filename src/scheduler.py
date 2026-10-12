"""
Pipeline Scheduler — automates the full workflow to run on its own,
without manual intervention. This is the "data engineering" piece:
new data in -> pipeline re-runs -> model retrains -> forecast refreshes.

Two ways to use this:

1. RUN AS A LONG-LIVED PROCESS (good for a demo):
   python src/scheduler.py
   Runs once immediately, then automatically every RUN_INTERVAL_HOURS.

2. RUN ONCE VIA CRON (good for production on a real server):
   Add this line to your crontab (`crontab -e`):
     0 2 * * *  cd /path/to/electricity-forecast && python3 src/scheduler.py --once
   This runs the full pipeline once, every day at 2 AM.

Each run:
  1. Re-runs pipeline.py   -> re-cleans + re-engineers features
  2. Re-runs train_model.py -> retrains and re-evaluates all 3 models
  3. Re-runs forecast.py    -> produces a fresh next-24h forecast
  4. Only keeps the new model if it performs at least as well as the
     previous one (a simple safety check — see `model_is_better`)
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import pipeline
import train_model
import forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [scheduler]  %(message)s")
log = logging.getLogger("scheduler")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METRICS_PATH = DATA_DIR / "metrics.csv"
METRICS_BACKUP = DATA_DIR / "metrics_previous.csv"
MODEL_PATH = DATA_DIR / "model.json"
MODEL_BACKUP = DATA_DIR / "model_previous.json"

RUN_INTERVAL_HOURS = 24  # how often to re-run when used as a long-lived process


def model_is_better(new_metrics_path, old_metrics_path) -> bool:
    """
    Compares the new XGBoost MAPE against the previous run's XGBoost MAPE.
    Returns True if the new model is at least as accurate (lower MAPE).
    If there's no previous run to compare against, accepts the new model.
    """
    if not old_metrics_path.exists():
        return True

    import pandas as pd
    new_m = pd.read_csv(new_metrics_path).set_index("model")
    old_m = pd.read_csv(old_metrics_path).set_index("model")

    new_mape = new_m.loc["XGBoost", "MAPE_%"]
    old_mape = old_m.loc["XGBoost", "MAPE_%"]

    log.info(f"New XGBoost MAPE: {new_mape}%  |  Previous: {old_mape}%")
    return new_mape <= old_mape


def run_once():
    log.info("=== Pipeline run starting ===")

    # Backup current model/metrics before overwriting, so we can compare
    if MODEL_PATH.exists():
        shutil.copy(MODEL_PATH, MODEL_BACKUP)
    if METRICS_PATH.exists():
        shutil.copy(METRICS_PATH, METRICS_BACKUP)

    log.info("Step 1/3: Running data pipeline...")
    pipeline.run()

    log.info("Step 2/3: Training and evaluating models...")
    train_model.run()

    if model_is_better(METRICS_PATH, METRICS_BACKUP):
        log.info("New model performs at least as well — keeping it.")
    else:
        log.warning("New model is WORSE than previous — rolling back model.json.")
        if MODEL_BACKUP.exists():
            shutil.copy(MODEL_BACKUP, MODEL_PATH)

    log.info("Step 3/3: Generating next-24h forecast...")
    forecast.run(hours_ahead=24)

    log.info("=== Pipeline run complete ===\n")


def run_forever():
    while True:
        run_once()
        log.info(f"Sleeping for {RUN_INTERVAL_HOURS} hours until next run...")
        time.sleep(RUN_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                         help="Run the pipeline a single time and exit (use this for cron)")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_forever()
