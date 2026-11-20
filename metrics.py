"""
Best-effort metrics export to a Prometheus Pushgateway.

The university pipeline exposes a Pushgateway at PUSHGATEWAY_URL, whose metrics
are scraped into Grafana. Locally (no PUSHGATEWAY_URL) every call here is a
no-op, so the training scripts behave identically on a laptop and on the server.

Usage:
    from metrics import push_metrics
    push_metrics("train_top_features",
                 {"best_roc_auc": 0.93, "best_accuracy": 0.91},
                 labels={"model": "xgboost"})
"""
import os


def _gateway(url: str) -> str:
    """Return the gateway base URL, keeping any path prefix.

    Our Pushgateway is served under a path prefix (e.g.
    ``http://pushgateway:9091/pushgateway``). prometheus_client appends
    ``/metrics/job/...`` to whatever we pass, so we must keep the full URL
    INCLUDING that prefix — stripping it to host:port would POST to the wrong
    endpoint and silently drop the metrics. We only ensure a scheme is present.
    """
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def push_metrics(job: str, metrics: dict, labels: dict | None = None) -> bool:
    """Push a flat dict of float metrics to the Pushgateway.

    Returns True on success, False if skipped (no URL) or on any failure —
    metrics export must never break the training pipeline.
    """
    url = os.getenv("PUSHGATEWAY_URL")
    if not url:
        print("[metrics] PUSHGATEWAY_URL not set — skipping metrics push.")
        return False

    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        labels = labels or {}
        label_names = list(labels.keys())

        for name, value in metrics.items():
            gauge = Gauge(name, f"{job} metric: {name}", label_names, registry=registry)
            (gauge.labels(**labels) if label_names else gauge).set(float(value))

        gateway = _gateway(url)
        push_to_gateway(gateway, job=job, registry=registry)
        print(f"[metrics] Pushed {len(metrics)} metric(s) to {gateway} (job={job}).")
        return True
    except Exception as exc:  # noqa: BLE001 — never fail the pipeline on metrics
        print(f"[metrics] Push skipped ({type(exc).__name__}: {exc}).")
        return False


def push_model_metrics(job: str, results: dict, prefix: str = "model_") -> bool:
    """Push metrics for EVERY model in one shot, tagged with a ``model`` label.

    ``results`` is ``{model_name: {metric_name: value}}`` (exactly the shape
    train_top_features.py already builds). All models go into a single registry
    and a single ``push_to_gateway`` call, because each push REPLACES everything
    under the job — pushing per-model in a loop would leave only the last one.

    Produces series like ``model_roc_auc{model="lightgbm"}`` so Grafana can
    compare all algorithms side by side. No-op locally (no PUSHGATEWAY_URL).
    """
    url = os.getenv("PUSHGATEWAY_URL")
    if not url:
        print("[metrics] PUSHGATEWAY_URL not set — skipping metrics push.")
        return False

    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()

        # One Gauge per metric name, with a single "model" label dimension.
        metric_names = {m for metrics in results.values() for m in metrics}
        gauges = {
            name: Gauge(f"{prefix}{name}", f"{job} {name} per model", ["model"], registry=registry)
            for name in metric_names
        }
        for model_name, metrics in results.items():
            for name, value in metrics.items():
                gauges[name].labels(model=model_name).set(float(value))

        gateway = _gateway(url)
        push_to_gateway(gateway, job=job, registry=registry)
        print(f"[metrics] Pushed {len(metric_names)} metric(s) x {len(results)} "
              f"model(s) to {gateway} (job={job}).")
        return True
    except Exception as exc:  # noqa: BLE001 — never fail the pipeline on metrics
        print(f"[metrics] Push skipped ({type(exc).__name__}: {exc}).")
        return False
