"""
Process 4 — Train tree-based models on the top-K ranked features, with MLflow.

Loads the Feast saved dataset, selects the top-K features from
feature_ranking.csv (produced by rank_features.py), then trains
RandomForest, GradientBoosting, XGBoost, and LightGBM — each with
hyperparameters found by Optuna tuning (tune_models.py). Every model is
tracked in MLflow (params + metrics + the fitted model artifact) and dumped
as a .joblib so predict_online.py can load it by name.

Run rank_features.py first so feature_ranking.csv exists.
"""
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    DROP_COLS,
    EXPERIMENT_NAME,
    FEAST_REPO_PATH,
    FEATURE_RANKING_CSV,
    IS_REMOTE,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    REGISTERED_MODEL_NAME,
    SAVED_DATASET_NAME,
    TARGET,
    TOP_K,
    load_training_dataframe,
)
from metrics import push_model_metrics

# Minimum ROC-AUC gain a candidate must beat the champion by to be promoted.
# 0.0 promotes on any improvement; set e.g. 0.001 to ignore noise.
PROMOTION_MARGIN = float(os.getenv("PROMOTION_MARGIN", "0.0"))


def build_models():
    """Name -> (estimator, params logged to MLflow)."""
    return {
        # Tuned params from mlruns/.../c917d318b612 (Optuna, 50 trials)
        "random_forest": (
            RandomForestClassifier(
                n_estimators=286,
                max_depth=27,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features="log2",
                random_state=42,
                n_jobs=-1,
            ),
            {"n_estimators": 286, "max_depth": 27, "min_samples_split": 10,
             "min_samples_leaf": 1, "max_features": "log2", "tuned": True},
        ),
        # No tuned params available; sensible defaults (GBM gains little from
        # deeper tuning on this dataset — the other tree models already saturate).
        "gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
             "subsample": 0.8},
        ),
        # Tuned params from mlruns/.../942dfaba3603 (Optuna, 50 trials)
        "xgboost": (
            XGBClassifier(
                n_estimators=792,
                max_depth=12,
                learning_rate=0.0999,
                subsample=0.718,
                colsample_bytree=0.820,
                min_child_weight=1,
                gamma=1.522,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            {"n_estimators": 792, "max_depth": 12, "learning_rate": 0.0999,
             "subsample": 0.718, "colsample_bytree": 0.820, "min_child_weight": 1,
             "gamma": 1.522, "tuned": True},
        ),
        # Tuned params from mlruns/.../56d258a0d7fd (Optuna, 50 trials)
        "lightgbm": (
            LGBMClassifier(
                n_estimators=990,
                max_depth=11,
                num_leaves=151,
                learning_rate=0.0242,
                subsample=0.640,
                colsample_bytree=0.834,
                min_child_samples=15,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            {"n_estimators": 990, "max_depth": 11, "num_leaves": 151,
             "learning_rate": 0.0242, "subsample": 0.640,
             "colsample_bytree": 0.834, "min_child_samples": 15, "tuned": True},
        ),
    }


def load_top_features():
    if not os.path.exists(FEATURE_RANKING_CSV):
        raise FileNotFoundError(
            f"{FEATURE_RANKING_CSV} not found — run rank_features.py first."
        )
    ranks = pd.read_csv(FEATURE_RANKING_CSV, index_col=0)
    return ranks.head(TOP_K).index.tolist()


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_and_log(name, model, params, top_features, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("top_features", ", ".join(top_features))
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("feast_dataset", SAVED_DATASET_NAME)
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        out_path = os.path.join(FEAST_REPO_PATH, f"model_{name}.joblib")
        joblib.dump(model, out_path)
        mlflow.log_artifact(out_path)

        print(f"    {name:20s} acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}  ->  {out_path}")
        # Return metrics + the run id so the champion can be registered later.
        return metrics, run.info.run_id


def _current_champion_auc(client):
    """ROC-AUC of the currently-promoted champion, or None on the first run."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        if "roc_auc" in mv.tags:                       # fast path: read the tag
            return float(mv.tags["roc_auc"])
        run = client.get_run(mv.run_id)                # fallback: the source run
        return float(run.data.metrics["roc_auc"])
    except Exception:                                  # noqa: BLE001 — no champion yet
        return None


def register_with_promotion_gate(ranked, run_ids):
    """Champion-Challenger promotion gate (MLflow Model Registry).

    The best model of this run is the *candidate*. It is registered as a new
    version, then promoted to @champion ONLY if it beats the current champion's
    ROC-AUC by PROMOTION_MARGIN; otherwise the champion is kept and the
    candidate is tagged @challenger (the 20% canary served by api.py). On
    promotion the previous champion is demoted to @challenger (rollback path).

    The registry needs a database-backed tracking server (the cluster MLflow),
    so this is skipped on the local file store. Never fails the pipeline.
    """
    if not IS_REMOTE:
        print("[register] Local file store has no Model Registry — skipping "
              "(runs on the server with MLFLOW_TRACKING_URI=http://...).")
        return
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        cand_name, cand_metrics = ranked[0]            # best of this run
        cand_auc = cand_metrics["roc_auc"]

        # Register the candidate and tag it for future comparison/auditability.
        mv = mlflow.register_model(
            model_uri=f"runs:/{run_ids[cand_name]}/model",
            name=REGISTERED_MODEL_NAME,
        )
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "roc_auc", str(cand_auc))
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "model_type", cand_name)

        champ_auc = _current_champion_auc(client)

        if champ_auc is None or cand_auc > champ_auc + PROMOTION_MARGIN:
            # Promote: demote the old champion to challenger, then move @champion.
            try:
                old = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", old.version)
            except Exception:                          # noqa: BLE001 — first promotion
                pass
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
            prev = "none" if champ_auc is None else f"{champ_auc:.4f}"
            print(f"[register] PROMOTED v{mv.version} ({cand_name}, auc={cand_auc:.4f}) "
                  f"-> @champion (previous champion auc={prev}).")
        else:
            # Keep champion; candidate becomes the challenger (20% canary).
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", mv.version)
            print(f"[register] kept champion (auc={champ_auc:.4f}); v{mv.version} "
                  f"({cand_name}, auc={cand_auc:.4f}) set @challenger.")
    except Exception as exc:  # noqa: BLE001 — registration must not break training
        print(f"[register] Registration skipped ({type(exc).__name__}: {exc}).")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_dataframe()
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS, axis=1)

    top_features = load_top_features()
    print(f"[train_top_features] Top {TOP_K} features: {top_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X[top_features], y, stratify=y, random_state=42
    )

    results = {}
    run_ids = {}
    for name, (model, params) in build_models().items():
        results[name], run_ids[name] = train_and_log(
            name, model, params, top_features, X_train, X_test, y_train, y_test
        )

    print("\n=== Summary (sorted by ROC-AUC) ===")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
    for name, m in ranked:
        print(f"{name:20s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}")

    # Champion-Challenger promotion gate: promote the best candidate to
    # @champion only if it beats the current champion; else set it @challenger.
    register_with_promotion_gate(ranked, run_ids)

    # Push EVERY model's metrics to the Pushgateway (Grafana), each tagged with
    # a `model` label so the dashboards compare algorithms. No-op locally.
    push_model_metrics(job=EXPERIMENT_NAME, results=results)


if __name__ == "__main__":
    main()
