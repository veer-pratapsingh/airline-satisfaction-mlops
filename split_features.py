"""
Process 1 — Split the airline feature table into predictor and target files.

Mirrors the first cells of feast.ipynb: load `airline_features.parquet`,
drop the label to build the predictor frame, and keep the label plus the
Feast keys (passenger_id, event_timestamp) for the target frame. Both are
written back as Parquet for the Feast FileSources to read.
"""
import pandas as pd

from config import (
    AIRLINE_FEATURES,
    ENTITY_KEY,
    PREDICTOR_DATA,
    S3_STORAGE_OPTIONS,
    TARGET,
    TARGET_DATA,
)


def main():
    print("[split_features] Loading", AIRLINE_FEATURES)
    df = pd.read_parquet(AIRLINE_FEATURES)
    print(f"    {df.shape[0]:,} rows x {df.shape[1]} columns")

    predictor_df = df.drop(columns=[TARGET])
    target_df = df[[TARGET, ENTITY_KEY, "event_timestamp"]]

    # storage_options is None locally (plain file write) and points pandas at
    # Garage on the server, so PREDICTOR_DATA/TARGET_DATA may be s3:// URIs.
    predictor_df.to_parquet(PREDICTOR_DATA, index=False, storage_options=S3_STORAGE_OPTIONS)
    target_df.to_parquet(TARGET_DATA, index=False, storage_options=S3_STORAGE_OPTIONS)

    print(f"    Saved predictor -> {PREDICTOR_DATA} ({predictor_df.shape[1]} cols)")
    print(f"    Saved target    -> {TARGET_DATA} ({target_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
