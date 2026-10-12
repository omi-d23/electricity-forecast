# Electricity Consumption Forecasting

Mini project (BE IT, Data Science). Forecasts hourly electricity load using
historical usage patterns, with a complete pipeline, three compared models,
genuine forward forecasting, automated retraining, and a live dashboard.

## Status: Phase 1-6 complete

Pipeline, feature engineering, modeling, dashboard, forward forecasting, and
pipeline automation are all done and working. Remaining: live MQTT ingestion
(prototype script included), a 4th model (Prophet/LSTM), and public deployment.

## Dataset

**UCI ElectricityLoadDiagrams20112014** — hourly electricity load (kW) for
a client, over ~3 years. Public benchmark dataset used widely in
forecasting research. Raw file: `data/electricity_raw.csv` (321 clients;
we use one client's series as the primary target — easy to extend to
multiple clients later).

## Project structure

```
src/pipeline.py        -> cleans raw data, resamples to hourly, engineers
                           features (cyclical time encodings, lags, rolling
                           stats), saves data/processed.csv
src/train_model.py      -> time-based train/test split (last 14 days held out),
                           trains Naive baseline, Linear Regression, XGBoost,
                           saves metrics + predictions for the dashboard
src/forecast.py         -> genuine forward forecasting: recursively predicts
                           the next N hours BEYOND the dataset (not just
                           backtesting on known data)
src/scheduler.py        -> automates pipeline -> train -> forecast as one
                           scheduled job, with a safety check that only
                           keeps a retrained model if it's at least as
                           accurate as the previous one
src/mqtt_ingest.py      -> prototype ingestion service for connecting to
                           live smart meter data over MQTT (not yet wired
                           into the main pipeline — see "Future scope")
app.py                  -> Streamlit dashboard: forward forecast, actual vs
                           predicted, model comparison, feature importance,
                           usage patterns
```

## Results (held-out last 14 days)

| Model              | MAE  | RMSE | MAPE  |
|--------------------|------|------|-------|
| Naive (lag-24h)     | 4.36 | 7.07 | 4.37% |
| Linear Regression   | 3.31 | 4.67 | 3.32% |
| **XGBoost**         | **2.99** | **4.37** | **3.04%** |

XGBoost improves ~30% over the naive baseline. Most important features:
`lag_1h` (last hour's load), `lag_168h` (same hour last week), `lag_24h`
(same hour yesterday) — confirms strong daily and weekly seasonality.

## How to run

In PowerShell, from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. The generated
files needed by the dashboard are already included in `data/`.

To rebuild the data and forecast, first download the raw dataset if it is not
already present, then run:

```powershell
python src/download_data.py
python src/pipeline.py
python src/train_model.py
python src/forecast.py --hours 24
```

## Publish to GitHub and deploy

The raw dataset is about 92 MB and is not needed to run the dashboard, so it is
excluded from Git. The processed data, model results, and forecast files needed
by `app.py` are included.

1. Install [Git for Windows](https://git-scm.com/download/win) if Git is not
   already installed.
2. On GitHub, create a **public** empty repository (for example,
   `electricity-forecast`). Do not initialize it with a README or other files.
3. In PowerShell, change to this project folder and run the following. Replace
   `YOUR-USERNAME` with your GitHub username:

   ```powershell
   git init
   git add .
   git commit -m "Initial project upload"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/electricity-forecast.git
   git push -u origin main
   ```

4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in
   with GitHub. Select **Create app**, choose the `electricity-forecast`
   repository and `main` branch, set the app file path to `app.py`, and deploy.

### Run the whole thing automatically

```bash
python src/scheduler.py --once      # runs pipeline -> train -> forecast, once
python src/scheduler.py             # same, but repeats every 24 hours (long-running)
```

For real automation on a server, use cron instead of the long-running mode:
```
0 2 * * *  cd /path/to/electricity-forecast && python3 src/scheduler.py --once
```

## How forward forecasting works

`train_model.py` only *evaluates* the model on data we already have (the
last 14 days) — useful for measuring accuracy, but not an actual forecast.

`forecast.py` does genuine forecasting: starting from the last known hour,
it predicts one hour ahead, then feeds that prediction back in as a lag
feature to predict the next hour, and so on — this is called **recursive
forecasting**. The dashboard's top section shows this live.

## How automation works

`scheduler.py` chains the three steps together and adds one safety check:
after retraining, it compares the new model's MAPE against the previous
run's MAPE, and only keeps the new model if it's at least as accurate —
otherwise it rolls back automatically. This is the kind of check a real
production pipeline needs, not just re-running blindly.

## Future scope

- Wire `mqtt_ingest.py` into the main pipeline so it reads from live meter
  data instead of a static CSV (see the script's docstring for the plan)
- Add Prophet or an LSTM as a 4th model for comparison
- Move from Cron/long-running mode to Apache Airflow for a production-style
  DAG with retries, logging, and a monitoring UI
- Deploy the dashboard publicly (Streamlit Community Cloud)
- Extend forecasting to multiple electricity clients at once
