# src/models/xgboost_model/model_trainer.py
import numpy as np
import xgboost as xgb
import os
import datetime
import joblib  # for sklearn models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from .config import MODEL_CONFIG, TRAINING_CONFIG
from src.pipelines.data_setup import FeatureConfig, DEFAULT_FEATURE_CONFIG
from src.pipelines.experiment_pipelines import build_feature_engineering_pipeline, build_preprocessor

# --- MLflow opcional y utilidades ---
os.environ["MLFLOW_ENABLE_LOGGED_MODELS"] = "false"

# MLflow opcional
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return v

class ModelTrainer:
    """
    Trains, evaluates, and validates an XGBoost regression model.
    """

    def __init__(self, model_params=None, training_params=None,
             use_mlflow: bool = True,
             mlflow_experiment: str | None = None,
             mlflow_tracking_uri: str | None = None,
             registered_model_name: str | None = None,
             tags: dict | None = None):
        self.model_params = model_params or MODEL_CONFIG
        self.training_params = training_params or TRAINING_CONFIG
        self.model = xgb.XGBRegressor(**self.model_params)
        # ---- Opciones MLflow ----
        self.use_mlflow = bool(use_mlflow and _MLFLOW_AVAILABLE)
        self.mlflow_experiment = (
            mlflow_experiment
            or os.getenv("EXPERIMENT_NAME")
            or os.getenv("MLFLOW_EXPERIMENT_NAME", "steel-energy")
        )
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.registered_model_name = registered_model_name
        self.tags = tags or {"model_type": "xgboost"}

        if self.use_mlflow and self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def _mlflow_start(self, run_name: str | None = None):
        if not self.use_mlflow:
            return None
        if mlflow.active_run() is not None:
            return mlflow.active_run()
        mlflow.set_experiment(self.mlflow_experiment)
        return mlflow.start_run(run_name=run_name)

    def _mlflow_log_params(self):
        if not self.use_mlflow:
            return
        # Hiperparámetros reales del modelo
        try:
            mlflow.log_params({f"model__{k}": v for k, v in self.model.get_params().items()})
        except Exception:
            mlflow.log_params({f"model__{k}": v for k, v in (self.model_params or {}).items()})
        # Parámetros de training
        if self.training_params:
            mlflow.log_params({f"train__{k}": v for k, v in self.training_params.items()})

    def _mlflow_log_metrics(self, metrics: dict, prefix: str = ""):
        if not self.use_mlflow:
            return
        safe = {f"{prefix}{k}": _to_float(v) for k, v in metrics.items()}
        mlflow.log_metrics(safe)

    def _mlflow_log_cv(self, scores, scorer_name: str = "r2"):
        if not self.use_mlflow:
            return
        scores = np.asarray(scores, dtype=float)
        mlflow.log_metric(f"cv_{scorer_name}_mean", _to_float(scores.mean()))
        mlflow.log_metric(f"cv_{scorer_name}_std", _to_float(scores.std()))
        for i, s in enumerate(scores, 1):
            mlflow.log_metric(f"cv_{scorer_name}_fold_{i}", _to_float(s))

    def _mlflow_log_artifacts_and_model(self, saved_path: str, model_type: str):
        if not self.use_mlflow:
            return
        # 1) Sube el .pkl versionado
        try:
            mlflow.log_artifact(saved_path)
        except Exception as e:
            print(f"[WARN] No se pudo loguear artifact .pkl: {e}")

        # 2) Guarda temporalmente el modelo en formato MLflow y súbelo como artifacts
        try:
            from tempfile import TemporaryDirectory
            from pathlib import Path
            with TemporaryDirectory() as tmpdir:
                local_dir = Path(tmpdir) / f"{model_type}_mlflow_model"
                mlflow.sklearn.save_model(sk_model=self.model, path=str(local_dir))
                mlflow.log_artifacts(str(local_dir), artifact_path="model")
                print(f"[INFO] Modelo MLflow subido como artifacts desde: {local_dir}")
        except Exception as e:
            print(f"[WARN] No se pudo guardar/subir modelo MLflow temporal: {e}")

    def _plot_residuals(self, y_true, y_pred, out_path: str):
        import matplotlib.pyplot as plt
        resid = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, resid, s=8)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicción")
        plt.ylabel("Residual")
        plt.title("XGBoost - Residuales")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    def train(self, X_train, y_train):
        """
        Train the XGBoost model on the given data.
        """
        print("[INFO] Training XGBoost model...")
        self.model.fit(X_train, y_train)
        print("[INFO] Training complete.")
        return self.model

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Evaluate model performance on training and test sets.
        """
        print("[INFO] Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2_test": r2_score(y_test, y_pred),
            "R2_train": r2_score(y_train, y_train_pred),
        }

        print("[INFO] Model Evaluation:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        return metrics

    def cross_validate(self, X, y):
        """
        Perform k-fold cross-validation on the training data.
        """
        print("[INFO] Running cross-validation...")
        scores = cross_val_score(
            self.model, X, y,
            scoring="r2",
            cv=self.training_params.get("cv_folds", 5)
        )
        print(f"[INFO] CV R² mean: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores
       
    def save_model(self, model_type="xgboost", timestamp=None):
        """
        Save model artifact under a unique versioned filename only.
        No 'current' duplicate is created.
        """
        import os, datetime, joblib

        # Timestamp for versioning
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Versioned directory for this model type
        versioned_dir = f"models/{model_type}/artifacts"
        os.makedirs(versioned_dir, exist_ok=True)

        # File path (unique)
        versioned_model_path = os.path.join(versioned_dir, f"model_{timestamp}.pkl")

        # Save model (sklearn joblib)
        joblib.dump(self.model, versioned_model_path)

        print(f"[INFO] ✅ Saved versioned model to: {versioned_model_path}")
        return versioned_model_path


    def run(self, X_train, X_test, y_train, y_test, model_type="xgboost", timestamp=None):
        """
        Full training + evaluation pipeline for XGBoost.
        Saves model with timestamped filename and logs metrics.
        """
        print("[INFO] Starting XGBoost training pipeline...")

        run_ctx = self._mlflow_start(run_name=f"{model_type}_run")
        try:
            # Params
            self._mlflow_log_params()

            # 1) Train
            self.train(X_train, y_train)

            # 2) Evaluate
            metrics = self.evaluate(X_train, X_test, y_train, y_test)
            self._mlflow_log_metrics(metrics)

            # 2.1) Residuales
            try:
                os.makedirs("reports/figures", exist_ok=True)
                y_pred = self.model.predict(X_test)
                resid_fig = "reports/figures/residuals_xgb.png"
                self._plot_residuals(y_test, y_pred, resid_fig)
                if self.use_mlflow:
                    mlflow.log_artifact(resid_fig)
            except Exception as e:
                print(f"[WARN] No se pudo generar/loguear residuales: {e}")

            # 3) Guardar modelo .pkl y subir artifacts (incluye formato MLflow temporal)
            saved_path = self.save_model(model_type=model_type, timestamp=timestamp)
            self._mlflow_log_artifacts_and_model(saved_path, model_type=model_type)
            # --- Guardar métricas para DVC + log a MLflow ---
            import json as _json
            metrics_path = "reports/metrics_xgb.json"
            os.makedirs("reports", exist_ok=True)
            with open(metrics_path, "w") as f:
                _json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
            if self.use_mlflow:
                mlflow.log_artifact(metrics_path)
            print(f"[INFO] XGBoost model saved at: {saved_path}")
            print(f"[INFO] XGBoost training pipeline complete.\n")
            return metrics

        finally:
            if self.use_mlflow and run_ctx and mlflow.active_run() and \
            mlflow.active_run().info.run_id == run_ctx.info.run_id:
                mlflow.end_run()


class PipelineModelTrainer(ModelTrainer):
    """
    Versión v2 del entrenador que encapsula la ingeniería de atributos y el
    estimador XGBoost dentro de un Pipeline de scikit-learn.
    """

    def __init__(
        self,
        model_params=None,
        training_params=None,
        *,
        feature_config: FeatureConfig | None = None,
        drop_na: bool = False,
        use_date_features: bool = True,
        use_mlflow: bool = True,
        mlflow_experiment: str | None = None,
        mlflow_tracking_uri: str | None = None,
        registered_model_name: str | None = None,
        tags: dict | None = None,
    ):
        super().__init__(
            model_params=model_params,
            training_params=training_params,
            use_mlflow=use_mlflow,
            mlflow_experiment=mlflow_experiment,
            mlflow_tracking_uri=mlflow_tracking_uri,
            registered_model_name=registered_model_name,
            tags=tags,
        )
        self.feature_config = feature_config or DEFAULT_FEATURE_CONFIG
        self.drop_na = drop_na
        self.use_date_features = use_date_features

        feature_engineering = build_feature_engineering_pipeline(
            self.feature_config,
            drop_na=self.drop_na,
            use_date_features=self.use_date_features,
        )
        preprocessor = build_preprocessor(
            self.feature_config,
            use_date_features=self.use_date_features,
        )
        estimator = xgb.XGBRegressor(**self.model_params)

        self.model = Pipeline(
            steps=[
                ("feature_engineering", feature_engineering),
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

        if self.use_mlflow:
            self.tags = {**(self.tags or {}), "pipeline": "sklearn", "pipeline_version": "v2"}
