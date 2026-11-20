"""
Process 3 — Build the point-in-time training dataset via Feast.

Mirrors the `get_historical_features` + `create_saved_dataset` cell in
feast.ipynb. Uses the target frame as the entity dataframe (it carries the
passenger_id, event_timestamp and label), joins the 27 predictor features at
those timestamps, and persists the result as a Feast SavedDataset so that
downstream training reads a stable, versioned snapshot.
"""
import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

from config import (
    FEAST_FEATURES,
    FEAST_REPO_PATH,
    S3_ENDPOINT_OVERRIDE,
    S3_STORAGE_OPTIONS,
    SAVED_DATASET_NAME,
    TARGET_DATA,
    TRAINING_DATASET_STORAGE,
)


def main():
    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    # storage_options is None locally; on the server it points pandas at Garage
    # (TARGET_DATA may be an s3:// URI).
    entity_df = pd.read_parquet(TARGET_DATA, storage_options=S3_STORAGE_OPTIONS)
    print(f"[create_training_dataset] entity rows: {len(entity_df):,}")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEAST_FEATURES,
    )

    # Diagnostic: confirm the point-in-time join actually matched rows.
    joined = training_df.to_df()
    print(f"[create_training_dataset] joined rows: {len(joined):,}  "
          f"cols: {len(joined.columns)}")

    dataset = store.create_saved_dataset(
        from_=training_df,
        name=SAVED_DATASET_NAME,
        storage=SavedDatasetFileStorage(
            TRAINING_DATASET_STORAGE, s3_endpoint_override=S3_ENDPOINT_OVERRIDE
        ),
        allow_overwrite=True,
    )
    print(f"    Saved dataset '{SAVED_DATASET_NAME}' -> {TRAINING_DATASET_STORAGE}")
    return dataset


if __name__ == "__main__":
    main()
