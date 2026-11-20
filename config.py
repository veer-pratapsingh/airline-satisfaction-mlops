"""
Shared configuration for the Feast MLOps pipeline.

All paths are absolute, anchored to this file's location (the project root),
so every script resolves the same files no matter which working directory
you launch it from.
"""
import os

# Project root = the folder that contains this config.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Feast repo layout -------------------------------------------------------
FEAST_REPO_PATH = os.path.join(PROJECT_ROOT, "feature_repo", "feature_repo")
DATA_DIR = os.path.join(FEAST_REPO_PATH, "data")
# Cleaned CSV produced by prepare_data.py / consumed by setup_feast.py
PREPARED_CSV = os.path.join(PROJECT_ROOT, "prepared_data", "airline_cleaned.csv")

# Source feature table produced by setup_feast.py (intermediate, always local)
AIRLINE_FEATURES = os.path.join(DATA_DIR, "airline_features.parquet")

# --- Object storage (Garage / S3) for Feast file data ------------------------
# On the server the CI/CD pipeline points these at Garage buckets:
#   FEAST_SOURCE_URI  = s3://feast      (offline feature parquets)
#   FEAST_DATASET_URI = s3://datasets   (saved training dataset)
# Locally they are unset and everything stays under feature_repo/.../data.
# AWS_ENDPOINT_URL is passed to Feast's FileSource/SavedDatasetFileStorage as
# s3_endpoint_override so its dask/pyarrow reads+writes reach Garage (not AWS).
FEAST_SOURCE_URI = os.getenv("FEAST_SOURCE_URI")
FEAST_DATASET_URI = os.getenv("FEAST_DATASET_URI")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # e.g. http://garage:3900
# Passed to Feast file objects; None locally so they behave as plain files.
S3_ENDPOINT_OVERRIDE = AWS_ENDPOINT_URL or None

# storage_options for pandas read_parquet/to_parquet against Garage (S3 API).
S3_STORAGE_OPTIONS = (
    {"client_kwargs": {"endpoint_url": AWS_ENDPOINT_URL}} if AWS_ENDPOINT_URL else None
)

# Split outputs (predictor = features, target = label) consumed by Feast.
if FEAST_SOURCE_URI:
    PREDICTOR_DATA = f"{FEAST_SOURCE_URI}/predictor_data.parquet"
    TARGET_DATA = f"{FEAST_SOURCE_URI}/target_data.parquet"
else:
    PREDICTOR_DATA = os.path.join(DATA_DIR, "predictor_data.parquet")
    TARGET_DATA = os.path.join(DATA_DIR, "target_data.parquet")

# SavedDatasetFileStorage location for the point-in-time training dataset.
# (Feast resolves a relative path against the repo; an s3:// path is absolute.)
if FEAST_DATASET_URI:
    TRAINING_DATASET_STORAGE = f"{FEAST_DATASET_URI}/training_dataset.parquet"
    # Absolute URI used to read the snapshot back directly with pandas.
    TRAINING_DATASET_READ_PATH = TRAINING_DATASET_STORAGE
else:
    TRAINING_DATASET_STORAGE = "data/training_dataset.parquet"
    TRAINING_DATASET_READ_PATH = os.path.join(DATA_DIR, "training_dataset.parquet")

SAVED_DATASET_NAME = "training_dataset"


def load_training_dataframe():
    """Load the Feast-created training snapshot directly with pandas.

    Feast's get_saved_dataset() re-reads the snapshot through a timestamp-
    filtered offline job that drops every row after an S3 round-trip (the
    event_timestamp tz/precision shifts outside the stored min/max window).
    The persisted parquet already holds the point-in-time-correct snapshot, so
    we read it straight from storage (local path or Garage via S3_STORAGE_OPTIONS).
    """
    import pandas as pd

    return pd.read_parquet(
        TRAINING_DATASET_READ_PATH, storage_options=S3_STORAGE_OPTIONS
    )

# --- Modelling ---------------------------------------------------------------
TARGET = "satisfaction"
# Non-feature columns carried alongside the features in the Feast dataset
DROP_COLS = [TARGET, "passenger_id", "event_timestamp"]
ENTITY_KEY = "passenger_id"

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURE_RANKING_CSV = os.path.join(PROJECT_ROOT, "feature_ranking.csv")
TOP_K = 10

# --- MLflow ------------------------------------------------------------------
# On the university server the CI/CD pipeline sets MLFLOW_TRACKING_URI to the
# in-cluster MLflow service (http://mlflow:5000). Locally we fall back to a
# file-based store under ./mlruns so the scripts still run on a laptop.
_LOCAL_MLFLOW = "file:" + os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _LOCAL_MLFLOW)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "airline_satisfaction_top_features")
# Name under which the champion model is registered in the MLflow Model Registry.
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "airline_satisfaction_champion")

# --- Server services (read from the CI/CD environment, optional locally) -----
# These are injected by .gitlab-ci.yml on the server. They are read lazily by
# the scripts that need them, so leaving them unset just keeps everything local.
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL")            # e.g. http://pushgateway:9091/pushgateway
FEAST_URL = os.getenv("FEAST_URL")                        # e.g. http://feast:6000

# True when running inside the server pipeline (MLflow points at a real server).
IS_REMOTE = MLFLOW_TRACKING_URI.startswith("http")

# --- Feature references ------------------------------------------------------
# The 27 predictor columns, in the order used throughout the notebook.
FEATURE_COLUMNS = [
    "age",
    "flight_distance",
    "seat_comfort",
    "departure_arrival_time_convenient",
    "food_and_drink",
    "gate_location",
    "inflight_wifi_service",
    "inflight_entertainment",
    "online_support",
    "ease_of_online_booking",
    "onboard_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "cleanliness",
    "online_boarding",
    "departure_delay",
    "arrival_delay",
    "total_service_score",
    "total_delay",
    "is_long_flight",
    "customer_type_loyal",
    "customer_type_disloyal",
    "travel_type_business",
    "travel_type_personal",
    "class_business",
    "class_eco",
    "class_eco_plus",
]

# Fully-qualified Feast feature references (used for historical + online reads)
FEAST_FEATURES = [f"predictor_features:{col}" for col in FEATURE_COLUMNS]
