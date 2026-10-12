"""
Downloads the raw UCI electricity dataset (~92MB) into data/electricity_raw.csv.
Run this once before src/pipeline.py if data/electricity_raw.csv is missing.
"""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/electricity/electricity.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "electricity_raw.csv"

if __name__ == "__main__":
    print(f"Downloading {URL} -> {OUT}")
    urllib.request.urlretrieve(URL, OUT)
    print("Done.")
