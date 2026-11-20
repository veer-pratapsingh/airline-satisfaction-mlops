"""
Process 5 — Load features into the online store for low-latency serving.

The notebook used `materialize_incremental(end_date=now)`, whose window
started only a few days back and therefore skipped the snapshot rows (whose
event_timestamp is a fixed past date), leaving the online store empty and
making online prediction return NaN.

Here we materialize an explicit, wide range so every entity's latest feature
values are written to the online store regardless of how old the snapshot is.
"""
import pandas as pd
from feast import FeatureStore

from config import FEAST_REPO_PATH

# Start well before the snapshot timestamp so all rows are picked up.
START_DATE = pd.Timestamp("2020-01-01", tz="UTC")


def main():
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    end_date = pd.Timestamp.now(tz="UTC")
    print(f"[materialize_online] Materializing {START_DATE} -> {end_date}")
    store.materialize(start_date=START_DATE, end_date=end_date)
    print("    Online store updated.")


if __name__ == "__main__":
    main()
