"""
Hyperparameter tuning for the tree models with Optuna + MLflow.

Loads the Feast saved dataset, selects the top-K ranked features, then runs
one Optuna study per tree model (XGBoost, LightGBM, RandomForest) optimizing
cross-validated ACCURACY. Each tuned model is evaluated on the held-out test
set, logged to MLflow, and dumped as a .joblib into the Feast repo so
predict_online.py can serve it.

Prereqs: run create_training_dataset.py and rank_features.py first.
"""
import os

import joblib
import mlflow
import mlflow.sklearn
import optuna
from feast import FeatureStore
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from config import (
    DROP_COLS,
    EXPERIMENT_NAME,
    FEAST_REPO_PATH,
    MLFLOW_TRACKING_URI,
    SAVED_DATASET_NAME,
    TARGET,
)
from train_top_features import load_top_features

N_TRIALS = 50
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --- search spaces: each returns a (params, estimator) pair for a trial -------
def suggest_xgb(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
    }
    return params, XGBClassifier(**params, eval_metric="logloss", random_state=42, n_jobs=-1)


def suggest_lgbm(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    return params, LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)


def suggest_rf(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    return params, RandomForestClassifier(**params, random_state=42, n_jobs=-1)


# name -> (suggest_fn, estimator_class, fixed kwargs for the refit model)
MODELS = {
    "xgboost":       (suggest_xgb, XGBClassifier, {"eval_metric": "logloss"}),
    "lightgbm":      (suggest_lgbm, LGBMClassifier, {"verbose": -1}),
    "random_forest": (suggest_rf, RandomForestClassifier, {}),
}


def tune_one(name, suggest_fn, est_cls, fixed_kwargs, X_train, X_test, y_train, y_test):
    def objective(trial):
        _, model = suggest_fn(trial)
        return cross_val_score(model, X_train, y_train,
                               cv=CV, scoring="accuracy", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=N_TRIALS)

    # refit the best params on the full training set, score on the test set
    best_model = est_cls(**study.best_params, random_state=42, n_jobs=-1, **fixed_kwargs)
    best_model.fit(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)

    with mlflow.start_run(run_name=f"{name}_tuned"):
        mlflow.log_param("model_type", name)
        mlflow.log_param("tuner", "optuna")
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("cv_best_accuracy", study.best_value)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(best_model, name="model")

        out = os.path.join(FEAST_REPO_PATH, f"model_{name}_tuned.joblib")
        joblib.dump(best_model, out)
        mlflow.log_artifact(out)

    print(f"    {name:14s} cv_acc={study.best_value:.4f}  test_acc={test_acc:.4f}  ->  {out}")
    return {"name": name, "cv_accuracy": study.best_value, "test_accuracy": test_acc,
            "best_params": study.best_params}


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    df = store.get_saved_dataset(SAVED_DATASET_NAME).to_df()
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS, axis=1)[load_top_features()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    print(f"[tune_models] train={X_train.shape}  test={X_test.shape}  trials/model={N_TRIALS}")

    results = []
    for name, (suggest_fn, est_cls, fixed_kwargs) in MODELS.items():
        results.append(
            tune_one(name, suggest_fn, est_cls, fixed_kwargs,
                     X_train, X_test, y_train, y_test)
        )

    print("\n=== Tuned results (sorted by test accuracy) ===")
    for r in sorted(results, key=lambda d: d["test_accuracy"], reverse=True):
        print(f"{r['name']:14s} cv_acc={r['cv_accuracy']:.4f}  test_acc={r['test_accuracy']:.4f}")

    best = max(results, key=lambda d: d["test_accuracy"])
    print(f"\nBest model: {best['name']}  (test_acc={best['test_accuracy']:.4f})")
    print(f"Best params: {best['best_params']}")


if __name__ == "__main__":
    main()
