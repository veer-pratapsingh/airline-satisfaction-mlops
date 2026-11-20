# Airline Passenger Satisfaction — MLOps Pipeline

> **Team project:** built by MLOps Team 1 for the MLOps course (VLBA) at
> Otto von Guericke University Magdeburg, summer semester 2026. It ran on the
> university's GitLab and server; this public copy of the final version is
> published by [Veer Pratap Singh](https://veer-pratap-singh-portfolio.vercel.app),
> a member of the team. Server login details and the raw dataset are not included
> (see [Data](#data)).

Predict whether an airline passenger is **satisfied** from their flight and
service-rating features, built as a reproducible MLOps pipeline using
**Feast** (feature store), **MLflow** (experiment tracking + model registry),
**Garage/S3** (object storage), **Postgres** (online store), and
**Prometheus Pushgateway → Grafana** (metrics). The whole thing runs on the
university server via **GitLab CI/CD** — you `git push`, the pipeline runs itself.

> See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the server interaction model in
> detail. No SSH shell is needed to run anything — pushing is enough.

---

## How the CI/CD pipeline works

The pipeline is defined in **`.gitlab-ci.yml`**. Every `git push` to the
university GitLab repo triggers a project-scoped **GitLab Runner** that executes
the stages inside the server's Docker network, reaching the platform services by
hostname (`mlflow:5000`, `feast:6000`, `garage:3900`, `pushgateway:9091`,
`postgres:5432`). Secrets (AWS keys, Postgres password) are injected as **masked
CI/CD variables**; non-secret endpoints are set in `.gitlab-ci.yml`.

It runs **four sequential stages**, passing artifacts downstream:

| Stage | Job | What it runs | Talks to |
|-------|-----|--------------|----------|
| `build-images` | `build-image` | `docker build` + push to the registry (layer-cached via `--cache-from`) | GitLab registry |
| `data` | `data` | `prepare_data → setup_feast → split_features → apply_features → create_training_dataset` | **Garage** (Feast registry + parquet) |
| `train` | `train` | `rank_features → train_top_features` | **MLflow** (tracking + registry, artifacts on Garage), **Pushgateway** → Grafana |
| `online-infer` | `online-infer` | `materialize_online → predict_online` | **Postgres** (Feast online store) |

Each Feast job first activates the server profile in `before_script`:
```yaml
cp feature_repo/feature_repo/feature_store.yaml.server feature_repo/feature_repo/feature_store.yaml
```

**Storage layout on the server:** Feast registry + offline parquet on
`s3://feast`, the saved training dataset on `s3://datasets`, the online store in
Postgres, and MLflow model artifacts on Garage. Garage is reached via the S3
`s3_endpoint_override` / `AWS_ENDPOINT_URL` (`http://garage:3900`).

> The same code runs locally with **no env vars** (MLflow → `./mlruns`, Feast →
> local SQLite, Pushgateway → no-op). `config.py` makes the scripts
> environment-aware, so nothing forks between laptop and server.

---

## Pipeline steps (what each script does)

```
prepare_data → setup_feast → split_features → apply_features →
create_training_dataset → rank_features → train_top_features →
materialize_online → predict_online
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `prepare_data.py` | Clean `airline.csv` → `prepared_data/airline_cleaned.csv` (impute, encode, cap outliers, engineer 3 features, scale). |
| 2 | `setup_feast.py` | Add entity key (`passenger_id`) + `event_timestamp`, Feast-safe names → `airline_features.parquet`. |
| 3 | `split_features.py` | Split into predictor (features) and target (label) parquet sources. |
| 4 | `apply_features.py` | `feast apply` — register entity + feature views into the Feast registry. |
| 5 | `create_training_dataset.py` | Point-in-time join (`get_historical_features`) → versioned Feast **saved dataset**. |
| 6 | `rank_features.py` | Rank features (RF importance + permutation + mutual info) → `feature_ranking.csv`. |
| 7 | `train_top_features.py` | Train 4 tuned tree models on the top-10 features; log to **MLflow**; **register champion/challenger** (promotion gate); push all model metrics to **Pushgateway → Grafana**. |
| 8 | `materialize_online.py` | Load latest features into the Feast online store (Postgres). |
| 9 | `predict_online.py` | Fetch online features for given passenger IDs and serve predictions (LightGBM). |

Run the whole pipeline locally with `python run_pipeline.py`.

---

## Champion–Challenger & REST serving

- **Model registry:** the train stage registers the best model as `@champion`
  and the second-best as `@challenger` in the MLflow Model Registry
  (`airline_satisfaction_champion`). A **promotion gate** only moves `@champion`
  to a new model if it beats the current champion's ROC-AUC; otherwise it stays
  and the candidate becomes `@challenger` (the old champion is the rollback target).
- **REST API (`api.py`):** a FastAPI service that loads `@champion`/`@challenger`
  and routes each request **80% champion / 20% challenger**, reporting which
  model served it. Run it on demand (not a CI stage — CI jobs are ephemeral):
  ```bash
  python api.py        # or: uvicorn api:app --host 0.0.0.0 --port 8000
  # docs at http://localhost:8000/docs
  curl http://localhost:8000/predict/129879
  ```

---

## Accessing the dashboards on the server (SSH tunnel)

The pipeline runs on the server; to **view the dashboards** in your browser,
open an SSH tunnel that just forwards ports (the `-N` flag = no remote shell):

```bash
ssh -N -L 80:localhost:80 -L 9000:localhost:9000 <team-user>@<course-server>
# login details come from the course; they are not part of this repository
```

Then open:
- **Grafana (model metrics):** http://localhost:80
- **Garage / S3 console:** http://localhost:9000

To also reach MLflow / a locally-run API, add more `-L` forwards, e.g.
`-L 5000:localhost:5000` (MLflow) or `-L 8000:localhost:8000` (the API).

---

## Data

The pipeline reads `airline.csv` from the repository root: about 130,000
passenger survey responses (the public *Airlines Customer Satisfaction* /
Invistico Airline survey data, available on Kaggle). It is not included in this
repository; download it and save it as `airline.csv`. The expected header is:

```
satisfaction,Customer Type,Age,Type of Travel,Class,Flight Distance,Seat comfort,
Departure/Arrival time convenient,Food and drink,Gate location,Inflight wifi service,
Inflight entertainment,Online support,Ease of Online booking,On-board service,
Leg room service,Baggage handling,Checkin service,Cleanliness,Online boarding,
Departure Delay in Minutes,Arrival Delay in Minutes
```

---

## Quick start (local)

```bash
python -m venv venv && source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python run_pipeline.py
```

Locally everything stays on disk: MLflow logs to `./mlruns`, Feast uses a local
SQLite online store, and the Pushgateway call is a no-op. Inspect runs with:

```bash
mlflow ui            # then open http://localhost:5000
```

---

## Repository layout

```
airline.csv                     # raw dataset (pipeline input, not committed; see Data)
config.py                       # shared, env-aware configuration
metrics.py                      # best-effort Pushgateway export (no-op locally)
prepare_data.py                 # data cleaning / feature engineering
setup_feast.py … predict_online.py   # pipeline steps (see table above)
tune_models.py                  # offline Optuna hyperparameter tuning
api.py                          # REST serving (champion/challenger)
run_pipeline.py                 # run all steps end-to-end locally
feature_repo/feature_repo/      # Feast repo
  feature_definitions.py        #   entities + feature views (s3_endpoint_override)
  feature_store.yaml            #   local profile (sqlite + file registry)
  feature_store.yaml.server     #   server profile (Postgres + Garage registry)
.gitlab-ci.yml                  # CI/CD pipeline (build → data → train → online-infer)
Dockerfile                      # pipeline image used by the runner
DEPLOYMENT.md                   # how to run on the university server
```

> Generated artifacts — `prepared_data/`, the Feast `data/` directory, and all
> `*.joblib` models — are git-ignored. The pipeline regenerates them and stores
> models in MLflow.

---

## Services

| Service | Local | Server (in-cluster) |
|---------|-------|---------------------|
| MLflow tracking + registry | `file:./mlruns` | `http://mlflow:5000` |
| Feast registry + offline store | local files | Garage S3 (`s3://feast`) |
| Feast online store | SQLite | Postgres |
| Saved training dataset | local file | Garage S3 (`s3://datasets`) |
| MLflow artifacts | local files | Garage S3 (`http://garage:3900`) |
| Metrics | no-op | Pushgateway → Grafana |

Dataset: airline passenger satisfaction survey (~130k rows, 27 features:
service ratings, flight distance, delays, customer/travel type, class).
