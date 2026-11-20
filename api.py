"""
REST API serving component for airline passenger satisfaction (Champion-Challenger).

Serves predictions by passenger_id using two models registered by the train
stage in the MLflow Model Registry:
    @champion   -> best model  (default ~80% of traffic)
    @challenger -> 2nd best    (~20% of traffic)
Each request is routed by a weighted split; the response says which model served.
Features come from the Feast online store via the Feast SDK (our pipeline's
registry — the shared Feast REST server runs a different project).

Model source:
    server (MLFLOW_TRACKING_URI=http://...) -> registry aliases @champion/@challenger
    local                                   -> model_lightgbm.joblib (champion only)

Config (env):
    CHAMPION_TRAFFIC   fraction routed to champion (default 0.8)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000     # or: python api.py
    interactive docs at http://localhost:8000/docs
"""
import os
import random

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    ENTITY_KEY,
    FEAST_FEATURES,
    FEAST_REPO_PATH,
    IS_REMOTE,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

LOCAL_MODEL_PATH = os.path.join(FEAST_REPO_PATH, "model_lightgbm.joblib")
CHAMPION_TRAFFIC = float(os.getenv("CHAMPION_TRAFFIC", "0.8"))


class _Model:
    """A loaded model plus how it was sourced and its expected feature order."""
    def __init__(self, estimator, source):
        self.estimator = estimator
        self.source = source
        self.feature_names = [str(c) for c in estimator.feature_names_in_]


class _State:
    champion = None
    challenger = None
    store = None


_state = _State()


def _load_alias(alias):
    """Load a registry model by alias (server only). Returns None on failure."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    estimator = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
    return _Model(estimator, f"registry:{REGISTERED_MODEL_NAME}@{alias}")


def _ensure_models():
    """Load champion (+ challenger) once. Registry on server, joblib locally."""
    if _state.champion is not None:
        return
    if IS_REMOTE:
        try:
            _state.champion = _load_alias("champion")
        except Exception as exc:  # noqa: BLE001
            print(f"[api] champion alias load failed ({exc}); using local joblib.")
        try:
            _state.challenger = _load_alias("challenger")
        except Exception as exc:  # noqa: BLE001 — challenger is optional
            print(f"[api] challenger alias not available ({exc}).")
    if _state.champion is None:
        est = joblib.load(LOCAL_MODEL_PATH)
        _state.champion = _Model(est, f"joblib:{os.path.basename(LOCAL_MODEL_PATH)}")


def _pick_model():
    """Weighted choice: champion with probability CHAMPION_TRAFFIC, else challenger."""
    if _state.challenger is not None and random.random() >= CHAMPION_TRAFFIC:
        return _state.challenger, "challenger"
    return _state.champion, "champion"


def _ensure_store():
    if _state.store is None:
        from feast import FeatureStore
        _state.store = FeatureStore(repo_path=FEAST_REPO_PATH)
    return _state.store


def _fetch(passenger_ids, feature_names):
    store = _ensure_store()
    features = store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=[{ENTITY_KEY: pid} for pid in passenger_ids],
    ).to_df()
    X = features[feature_names]
    if X.isna().any().any():
        raise HTTPException(
            status_code=404,
            detail="Features missing in the online store for one or more ids. "
                   "Has materialize_online.py run?",
        )
    return X


def _predict(passenger_ids, model=None, role=None):
    _ensure_models()
    if model is None:
        model, role = _pick_model()
    X = _fetch(passenger_ids, model.feature_names)
    preds = model.estimator.predict(X)
    proba = model.estimator.predict_proba(X)[:, 1]
    results = [
        {
            "passenger_id": int(pid),
            "satisfied": int(p),
            "label": "satisfied" if p else "dissatisfied",
            "probability": round(float(pr), 4),
            "served_by": role,
            "model": model.source,
        }
        for pid, p, pr in zip(passenger_ids, preds, proba)
    ]
    return results


app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="Champion-Challenger prediction by passenger_id.",
    version="2.0.0",
)


class PredictRequest(BaseModel):
    passenger_ids: list[int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    _ensure_models()
    return {
        "registered_name": REGISTERED_MODEL_NAME,
        "champion": _state.champion.source if _state.champion else None,
        "challenger": _state.challenger.source if _state.challenger else None,
        "champion_traffic": CHAMPION_TRAFFIC,
        "features": _state.champion.feature_names if _state.champion else None,
    }


@app.get("/predict/{passenger_id}")
def predict_one(passenger_id: int):
    """Predict for one passenger; model chosen by the champion/challenger split."""
    return _predict([passenger_id])[0]


@app.post("/predict")
def predict_many(req: PredictRequest):
    """Predict for a batch; one model (per the split) serves the whole batch."""
    if not req.passenger_ids:
        raise HTTPException(status_code=400, detail="passenger_ids must not be empty")
    return {"predictions": _predict(req.passenger_ids)}


@app.get("/predict/{passenger_id}/{role}")
def predict_with_role(passenger_id: int, role: str):
    """Force a specific model: role = 'champion' or 'challenger' (for A/B testing)."""
    _ensure_models()
    model = _state.champion if role == "champion" else _state.challenger
    if model is None:
        raise HTTPException(status_code=404, detail=f"No model for role '{role}'")
    return _predict([passenger_id], model=model, role=role)[0]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
