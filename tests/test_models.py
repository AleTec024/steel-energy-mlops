import joblib
import numpy as np
import pytest
import types
from pathlib import Path

from src.models.linear_regression_model import model_trainer as lin_mod
from src.models.random_forest_model import model_trainer as rf_mod
from src.models.xgboost_model import model_trainer as xgb_mod
from src.models.linear_regression_model.model_trainer import ModelTrainer as LinearTrainer
from src.models.random_forest_model.model_trainer import ModelTrainer as RandomForestTrainer
from src.models.xgboost_model.model_trainer import ModelTrainer as XGBTrainer


@pytest.fixture
def regression_data():
    # Simple linear relation: y = 2*x1 + 0.5*x2
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
            [6.0, 12.0],
            [7.0, 14.0],
            [8.0, 16.0],
        ]
    )
    y = 2 * X[:, 0] + 0.5 * X[:, 1]
    # Train on first 6, test on last 2 to keep tests quick
    return (X[:6], X[6:], y[:6], y[6:])


def assert_metrics_structure(metrics: dict):
    expected_keys = {"RMSE", "MAE", "R2_test", "R2_train"}
    assert expected_keys.issubset(metrics.keys())
    assert all(np.isfinite(list(metrics.values())))


@pytest.fixture
def stub_mlflow(monkeypatch, tmp_path):
    import mlflow

    state = {"uri": f"file://{tmp_path}", "experiment": None, "active": None}

    class DummyRun:
        def __init__(self, run_id="run-123"):
            self.info = types.SimpleNamespace(run_id=run_id)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            state["active"] = None

    def start_run(run_name=None):
        state["active"] = DummyRun()
        return state["active"]

    def active_run():
        return state.get("active")

    def end_run():
        state["active"] = None

    def set_tracking_uri(uri):
        state["uri"] = uri

    def get_tracking_uri():
        return state["uri"]

    def set_experiment(name):
        state["experiment"] = name

    def get_experiment_by_name(name):
        exp = state.setdefault("experiments", {})
        return exp.get(name)

    def create_experiment(name, artifact_location=None):
        exp = state.setdefault("experiments", {})
        exp[name] = types.SimpleNamespace(artifact_location=artifact_location)
        return name

    def log_params(params):
        state["params"] = params

    def log_metrics(metrics):
        state["metrics"] = metrics

    def log_metric(key, value):
        state.setdefault("metric_items", {})[key] = value

    def log_artifact(path, artifact_path=None):
        state.setdefault("artifacts", []).append(Path(path))

    def log_artifacts(path, artifact_path=None):
        state.setdefault("artifacts", []).append(Path(path))

    def register_model(model_uri, name):
        return types.SimpleNamespace(version=1)

    class DummyClient:
        def transition_model_version_stage(self, name, version, stage, archive_existing_versions=True):
            state.setdefault("transitions", []).append((name, version, stage))

    sklearn_ns = types.SimpleNamespace(
        save_model=lambda sk_model, path: Path(path).mkdir(parents=True, exist_ok=True),
        log_model=lambda sk_model, artifact_path="model": None,
    )
    xgb_ns = types.SimpleNamespace(log_model=lambda xgb_model, artifact_path="model": None)

    monkeypatch.setattr(mlflow, "start_run", start_run)
    monkeypatch.setattr(mlflow, "active_run", active_run)
    monkeypatch.setattr(mlflow, "end_run", end_run)
    monkeypatch.setattr(mlflow, "set_tracking_uri", set_tracking_uri)
    monkeypatch.setattr(mlflow, "get_tracking_uri", get_tracking_uri)
    monkeypatch.setattr(mlflow, "set_experiment", set_experiment)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", get_experiment_by_name)
    monkeypatch.setattr(mlflow, "create_experiment", create_experiment)
    monkeypatch.setattr(mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow, "log_metrics", log_metrics)
    monkeypatch.setattr(mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow, "log_artifacts", log_artifacts)
    monkeypatch.setattr(mlflow, "register_model", register_model)
    monkeypatch.setattr(mlflow, "sklearn", sklearn_ns, raising=False)
    monkeypatch.setattr(mlflow, "xgboost", xgb_ns, raising=False)
    monkeypatch.setattr(mlflow, "tracking", types.SimpleNamespace(MlflowClient=DummyClient))

    return state


def test_linear_trainer_train_evaluate_and_save(tmp_path, monkeypatch, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    monkeypatch.chdir(tmp_path)

    trainer = LinearTrainer(use_mlflow=False)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_train, X_test, y_train, y_test)
    assert_metrics_structure(metrics)

    path = trainer.save_model(model_type="linear_regression", timestamp="20240101")
    expected = tmp_path / "models" / "linear_regression" / "artifacts" / "model_20240101.pkl"
    assert Path(path).resolve() == expected
    assert joblib.load(path)


def test_linear_trainer_run_executes_with_stubbed_mlflow(tmp_path, monkeypatch, regression_data, stub_mlflow):
    X_train, X_test, y_train, y_test = regression_data
    monkeypatch.chdir(tmp_path)
    import src.utils.env as env_mod
    monkeypatch.setattr(env_mod, "load_env", lambda: {"MLFLOW_TRACKING_URI": "file://" + str(tmp_path), "EXPERIMENT_NAME": "exp"})
    monkeypatch.setattr(LinearTrainer, "_plot_residuals", lambda *_, **__: None)

    trainer = LinearTrainer(use_mlflow=True)
    metrics = trainer.run(X_train, X_test, y_train, y_test, timestamp="20240110")

    assert_metrics_structure(metrics)
    assert (tmp_path / "reports" / "metrics_linear.json").exists()
    assert (tmp_path / "models" / "linear_regression" / "artifacts" / "model_20240110.pkl").exists()


def test_random_forest_trainer_train_evaluate_and_save(tmp_path, monkeypatch, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    monkeypatch.chdir(tmp_path)

    trainer = RandomForestTrainer(
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        training_params={"cv_folds": 2, "test_size": 0.2},
        use_mlflow=False,
    )
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_train, X_test, y_train, y_test)
    assert_metrics_structure(metrics)

    path = trainer.save_model(model_type="random_forest", timestamp="20240102")
    expected = tmp_path / "models" / "random_forest" / "artifacts" / "model_20240102.pkl"
    assert Path(path).resolve() == expected
    assert joblib.load(path)


def test_random_forest_trainer_run_executes_with_stubbed_mlflow(tmp_path, monkeypatch, regression_data, stub_mlflow):
    X_train, X_test, y_train, y_test = regression_data
    monkeypatch.chdir(tmp_path)

    import src.utils.env as env_mod
    monkeypatch.setattr(env_mod, "load_env", lambda: {"MLFLOW_TRACKING_URI": "file://" + str(tmp_path), "EXPERIMENT_NAME": "exp"})
    monkeypatch.setattr(RandomForestTrainer, "_plot_residuals", staticmethod(lambda *_, **__: None))
    trainer = RandomForestTrainer(
        model_params={"n_estimators": 5, "random_state": 0, "n_jobs": 1},
        training_params={"cv_folds": 2, "test_size": 0.2},
        use_mlflow=True,
    )

    metrics = trainer.run(X_train, X_test, y_train, y_test, timestamp="20240111")

    assert_metrics_structure(metrics)
    assert (tmp_path / "reports" / "metrics_rf.json").exists()
    assert (tmp_path / "models" / "random_forest" / "artifacts" / "model_20240111.pkl").exists()


def test_xgb_trainer_train_evaluate_and_save(tmp_path, monkeypatch, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    monkeypatch.chdir(tmp_path)

    trainer = XGBTrainer(
        model_params={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "eval_metric": "rmse",
            "random_state": 0,
            "n_jobs": 1,
        },
        training_params={"cv_folds": 2, "test_size": 0.2},
        use_mlflow=False,
    )
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_train, X_test, y_train, y_test)
    assert_metrics_structure(metrics)

    path = trainer.save_model(model_type="xgboost", timestamp="20240103")
    expected = tmp_path / "models" / "xgboost" / "artifacts" / "model_20240103.pkl"
    assert Path(path).resolve() == expected
    assert joblib.load(path)


def test_xgb_trainer_run_executes_with_stubbed_mlflow(tmp_path, monkeypatch, regression_data, stub_mlflow):
    X_train, X_test, y_train, y_test = regression_data
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(XGBTrainer, "_plot_residuals", lambda *_, **__: None)
    trainer = XGBTrainer(
        model_params={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "eval_metric": "rmse",
            "random_state": 0,
            "n_jobs": 1,
        },
        training_params={"cv_folds": 2, "test_size": 0.2},
        use_mlflow=True,
    )

    metrics = trainer.run(X_train, X_test, y_train, y_test, timestamp="20240112")

    assert_metrics_structure(metrics)
    assert (tmp_path / "reports" / "metrics_xgb.json").exists()
    assert (tmp_path / "models" / "xgboost" / "artifacts" / "model_20240112.pkl").exists()
