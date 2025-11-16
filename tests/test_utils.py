import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils import env as env_mod
from src.utils import mlflow_smoke_test
from src.utils import seeds


def test_load_env_defaults_and_env_file(monkeypatch, tmp_path):
    # Ensure clean environment
    for key in ["ENV", "EXPERIMENT_NAME", "MLFLOW_TRACKING_URI", "BACKEND_URI", "ARTIFACTS_URI", "AWS_PROFILE", "AWS_REGION"]:
        monkeypatch.delenv(key, raising=False)

    # With .env present
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "ENV=prod\nEXPERIMENT_NAME=my-exp\nMLFLOW_TRACKING_URI=http://mlflow\nBACKEND_URI=db://uri\nARTIFACTS_URI=s3://bucket\nAWS_PROFILE=dev\nAWS_REGION=us-west-2\n"
    )
    values = env_mod.load_env()
    assert values["ENV"] == "prod"
    assert values["EXPERIMENT_NAME"] == "my-exp"
    assert values["MLFLOW_TRACKING_URI"] == "http://mlflow"
    assert values["BACKEND_URI"] == "db://uri"
    assert values["ARTIFACTS_URI"] == "s3://bucket"
    assert values["AWS_PROFILE"] == "dev"
    assert values["AWS_REGION"] == "us-west-2"

    # Without .env, falls back to environment variables
    (tmp_path / ".env").unlink()
    for key in ["ENV", "EXPERIMENT_NAME", "MLFLOW_TRACKING_URI", "BACKEND_URI", "ARTIFACTS_URI", "AWS_PROFILE", "AWS_REGION"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EXPERIMENT_NAME", "fallback-exp")
    values = env_mod.load_env()
    assert values["EXPERIMENT_NAME"] == "fallback-exp"
    assert values["ENV"] == "local"


def test_set_global_seed_is_reproducible():
    seeds.set_global_seed(123)
    a = (random.random(), np.random.rand())
    seeds.set_global_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_mlflow_smoke_test_runs_with_stubbed_mlflow(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Stub env loader to avoid reading disk or real env vars
    monkeypatch.setattr(mlflow_smoke_test, "load_env", lambda: {"MLFLOW_TRACKING_URI": "file://mlruns", "EXPERIMENT_NAME": "stub-exp"})

    state = {"tags": None, "params": None, "metrics": None, "artifacts": []}

    class DummyRun:
        def __init__(self):
            self.info = SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMLflow:
        def set_tracking_uri(self, uri):
            state["uri"] = uri

        def set_experiment(self, name):
            state["experiment"] = name

        def start_run(self):
            return DummyRun()

        def set_tags(self, tags):
            state["tags"] = tags

        def log_params(self, params):
            state["params"] = params

        def log_metrics(self, metrics):
            state["metrics"] = metrics

        def log_artifact(self, path):
            state["artifacts"].append(Path(path))

    monkeypatch.setattr(mlflow_smoke_test, "mlflow", DummyMLflow())

    mlflow_smoke_test.run_smoke_test()

    assert state["uri"] == "file://mlruns"
    assert state["experiment"] == "stub-exp"
    assert state["params"] == {"probe": "smoke_test"}
    assert state["metrics"] == {"mae": 1.23, "r2": 0.89}
    assert (tmp_path / "reports" / "smoke.txt").exists()
    assert state["artifacts"][0].resolve() == (tmp_path / "reports" / "smoke.txt")
