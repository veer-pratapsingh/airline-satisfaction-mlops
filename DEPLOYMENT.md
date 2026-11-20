# Running on the university server (no SSH command line needed)

This project runs on the university infrastructure **through GitLab CI/CD** —
not by logging into the server and typing commands. You interact with the
server in exactly three ways:

1. **`git push`** to the university GitLab repo → the GitLab Runner on the
   server automatically executes `.gitlab-ci.yml`.
2. The pipeline jobs run **inside the server's Docker network**, so they reach
   the services by hostname: `mlflow:5000`, `feast:6000`, `garage:3900`,
   `pushgateway:9091`.
3. **An SSH tunnel** just forwards ports so you can open the **dashboards in
   your browser** — it does not give (or need) a shell. Note the `-N` flag =
   "no remote command".

```
ssh -N -L 80:localhost:80 -L 9000:localhost:9000 <team-user>@<course-server>
# then open:  http://localhost:80   (Grafana)   http://localhost:9000  (S3/Garage console)
```
The team login comes from the course and is not part of this repository.

---

## One-time setup

### 1. Point the repo at the university GitLab

Your code is currently on GitHub. Add the GitLab project as a remote and push:

```bash
# keep GitHub as 'origin' if you like, add the uni GitLab as 'gitlab'
git remote add gitlab <UNIVERSITY_GITLAB_PROJECT_URL>.git
git add .
git commit -m "Add CI/CD pipeline, Docker, env-aware config"
git push gitlab main      # triggers the pipeline
```

(If you'd rather make GitLab the default: `git remote set-url origin <URL>.git`.)

### 2. Confirm the CI/CD variables exist

In **GitLab → Settings → CI/CD → Variables**, these must be present (your
instructor said they were added to your project — verify):

| Variable | Purpose | Set as |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Garage/S3 access (MLflow artifacts) | masked |
| `AWS_SECRET_ACCESS_KEY` | Garage/S3 secret | masked |
| `POSTGRES_FEAST_PASSWORD` | Feast Postgres online store (only if you enable the server profile) | masked |

The non-secret endpoints (`MLFLOW_TRACKING_URI`, `AWS_ENDPOINT_URL`,
`PUSHGATEWAY_URL`, …) are already hardcoded in `.gitlab-ci.yml`.

### 3. Make sure the raw data is committed

`prepare_data.py` reads `airline.csv` at the repo root. It must be tracked so
the runner has it:

```bash
git add airline.csv
```

> `prepared_data/` and `feature_repo/feature_repo/data/` are git-ignored on
> purpose — the pipeline regenerates them and passes them between jobs as CI
> **artifacts**.

---

## What the pipeline does

`.gitlab-ci.yml` defines four stages; each runs in the image built by the first:

| Stage | Job | Runs | Talks to |
|---|---|---|---|
| `build-images` | `build-image` | `docker build` + push to project registry | GitLab registry |
| `data` | `data` | `prepare_data → setup_feast → split_features → apply_features → create_training_dataset` | — (file/parquet) |
| `train` | `train` | `rank_features → train_top_features` | **MLflow** (tracking + artifacts on Garage/S3), **Pushgateway** (best-model metrics → Grafana) |
| `online-infer` | `online-infer` | `materialize_online → predict_online` | Feast online store |

Artifacts (parquet data, registry, models) flow downstream automatically.

---

## Local vs. server — same code, different env

The code is environment-aware (`config.py`, `metrics.py`), so nothing forks:

- **Locally** (no env vars): MLflow logs to `./mlruns`, Feast uses local
  sqlite, the Pushgateway call is a no-op. Run `python run_pipeline.py`.
- **On the server**: `.gitlab-ci.yml` sets `MLFLOW_TRACKING_URI=http://mlflow:5000`,
  the S3 endpoint, and `PUSHGATEWAY_URL`, so the same scripts log to the
  cluster services.

---

## Optional upgrade: Postgres-backed Feast online store

The default pipeline keeps Feast file/sqlite-based (simple, fully working).
To use the real Postgres online store + an S3 registry that persists across
jobs, switch to the server profile:

1. Add the Postgres connection CI variables: `POSTGRES_FEAST_HOST`,
   `POSTGRES_FEAST_PORT`, `POSTGRES_FEAST_DB`, `POSTGRES_FEAST_USER`
   (and `POSTGRES_FEAST_PASSWORD`, already present).
2. In each Feast job's `before_script`, activate the server profile:
   ```yaml
   before_script:
     - cp feature_repo/feature_repo/feature_store.yaml.server feature_repo/feature_repo/feature_store.yaml
   ```
   (See `feature_store.yaml.server` — registry on `s3://feast/registry.db`,
   online store on Postgres.)
3. With the registry on S3 you can also split `materialize_online` and
   `predict_online` into separate stages instead of one job.
