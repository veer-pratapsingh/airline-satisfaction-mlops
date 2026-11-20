"""
Process 6 — Serve predictions from the online feature store.

Mirrors the final cells of feast.ipynb: fetch the latest features for a set
of passenger_ids from the online store, then run a trained model on them.
The model is loaded by name and only the columns it was trained on
(`feature_names_in_`) are passed in, so this works for either the
top-feature LogisticRegression or RandomForest produced by
train_top_features.py.

Usage:
    python predict_online.py                      # default sample ids
    python predict_online.py 129879 129878 5      # custom passenger ids
"""
import os
import sys

import joblib
from feast import FeatureStore

from config import ENTITY_KEY, FEAST_FEATURES, FEAST_REPO_PATH

DEFAULT_IDS = [129879, 129878]
MODEL_NAME = "model_lightgbm.joblib"


def get_passenger_ids():
    if len(sys.argv) > 1:
        return [int(x) for x in sys.argv[1:]]
    return DEFAULT_IDS


def main():
    passenger_ids = get_passenger_ids()
    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    features = store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=[{ENTITY_KEY: pid} for pid in passenger_ids],
    ).to_df()

    model_path = os.path.join(FEAST_REPO_PATH, MODEL_NAME)
    model = joblib.load(model_path)

    X = features[model.feature_names_in_]
    if X.isna().any().any():
        raise ValueError(
            "Online features contain NaN — the online store is missing these "
            "entities. Run materialize_online.py and check the feature view TTL."
        )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print(f"[predict_online] model: {MODEL_NAME}")
    for pid, pred, prob in zip(passenger_ids, predictions, probabilities):
        print(f"    passenger {pid}: satisfied={pred}  p(satisfied)={prob:.4f}")


if __name__ == "__main__":
    main()
