"""Optional MLflow mirror.

Several sessions on this machine share one MLflow instance, so runs are logged
there for cross-project visibility -- but the source of truth stays local
(runs/<name>/log.jsonl + summary.json).  Every call is best-effort: a stopped
tracking server, a missing package or a network hiccup must never be able to
kill a training run that has been going for an hour.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_ENABLED = False
_mlflow = None


def init(experiment: str, run_name: str, params: Optional[Dict[str, Any]] = None) -> bool:
    """Start a run if MLFLOW_TRACKING_URI is set and the client is importable."""
    global _ENABLED, _mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return False
    try:
        import mlflow                     # noqa: PLC0415
        _mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        if params:
            flat = {k: v for k, v in params.items()
                    if isinstance(v, (int, float, str, bool)) or v is None}
            for i in range(0, len(flat), 100):        # the API caps params per call
                mlflow.log_params(dict(list(flat.items())[i:i + 100]))
        _ENABLED = True
        print(f"[mlflow] logging to {uri} as {experiment}/{run_name}", flush=True)
    except Exception as e:
        print(f"[mlflow] disabled ({type(e).__name__}: {str(e)[:80]})", flush=True)
        _ENABLED = False
    return _ENABLED


def log(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if not _ENABLED:
        return
    try:
        _mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                             if isinstance(v, (int, float))}, step=step)
    except Exception:
        pass


def log_artifact(path: str) -> None:
    if not _ENABLED:
        return
    try:
        _mlflow.log_artifact(path)
    except Exception:
        pass


def finish(status: str = "FINISHED") -> None:
    global _ENABLED
    if not _ENABLED:
        return
    try:
        _mlflow.end_run(status=status)
    except Exception:
        pass
    _ENABLED = False
