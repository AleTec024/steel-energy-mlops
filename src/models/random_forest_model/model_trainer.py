import joblib
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from .config import MODEL_CONFIG, TRAINING_CONFIG

# MLflow + utilidades
import json
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.env import load_env  # para cargar el .env
from src.pipelines.data_setup import FeatureConfig, DEFAULT_FEATURE_CONFIG

import os
os.environ["MLFLOW_ENABLE_LOGGED_MODELS"] = "false"

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

class ModelTrainer:
    """
    Trains, evaluates, and validates a Random Forest regression model.
    """

    def __init__(self, model_params=None, training_params=None,
             use_mlflow: bool = True,
             mlflow_experiment: str | None = None,
             mlflow_tracking_uri: str | None = None,
             registered_model_name: str | None = None,
             tags: dict | None = None):
        self.model_params = model_params or MODEL_CONFIG
        self.training_params = training_params or TRAINING_CONFIG
        self.model = RandomForestRegressor(**self.model_params)

        # ---- Opciones MLflow (igual que XGB) ----
        self.use_mlflow = bool(use_mlflow and _MLFLOW_AVAILABLE)
        self.mlflow_experiment = (
            mlflow_experiment
            or os.getenv("RF_EXPERIMENT_NAME")          # << permite experimento específico para RF
            or os.getenv("EXPERIMENT_NAME")
            or os.getenv("MLFLOW_EXPERIMENT_NAME", "steel-energy")
        )
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.registered_model_name = registered_model_name
        self.tags = tags or {"model_type": "random_forest"}

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
        try:
            mlflow.log_params({f"model__{k}": v for k, v in self.model.get_params().items()})
        except Exception:
            mlflow.log_params({f"model__{k}": v for k, v in (self.model_params or {}).items()})
        if self.training_params:
            mlflow.log_params({f"train__{k}": v for k, v in self.training_params.items()})

    def _mlflow_log_metrics(self, metrics: dict):
        if not self.use_mlflow:
            return
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def train(self, X_train, y_train):
        print("[INFO] Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("[INFO] Training complete.")
        return self.model

    def evaluate(self, X_train, X_test, y_train, y_test):
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
        print("[INFO] Running cross-validation...")
        scores = cross_val_score(self.model, X, y, scoring="r2",
                                 cv=self.training_params.get("cv_folds", 5))
        print(f"[INFO] CV R² mean: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    def save_model(self, model_type="random_forest", timestamp=None):
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

     # ---------------------- MLflow helpers ----------------------
    @staticmethod
    def _ensure_output_dirs():
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _plot_residuals(y_true, y_pred, out_path: str):
        resid = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, resid, s=8)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicción")
        plt.ylabel("Residual")
        plt.title("Random Forest - Residuales")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    def run(self, X_train, X_test, y_train, y_test, model_type="random_forest", timestamp=None):
        """
        Full training + evaluation pipeline for Random Forest.
        Saves model with timestamped filename and logs metrics.
        """
        # 0) Configurar entorno de MLflow
        env_vars = load_env()
        mlflow.set_tracking_uri(env_vars["MLFLOW_TRACKING_URI"])

        # Experimento específico para RF (p.ej., "steel-energy-rf")
        base_exp = env_vars.get("EXPERIMENT_NAME", "steel-energy")
        rf_exp = os.getenv("RF_EXPERIMENT_NAME", f"{base_exp}-rf")
        mlflow.set_experiment(rf_exp)

        self._ensure_output_dirs()
        print("[INFO] Starting Random Forest training pipeline...")

        with mlflow.start_run(run_name="random_forest_run"):
            # 1) Parámetros y tags
            mlflow.set_tags({
                "model_family": "random_forest",
                "stage": os.getenv("RUN_STAGE", "dev")
            })
            mlflow.log_params({f"model__{k}": v for k, v in self.model.get_params().items()})
            mlflow.log_params({f"train__{k}": v for k, v in self.training_params.items()})
            mlflow.log_params({
                "n_train": int(getattr(X_train, "shape", [len(X_train)])[0]),
                "n_test": int(getattr(X_test, "shape", [len(X_test)])[0]),
            })

            # 2) Entrenar
            self.train(X_train, y_train)

            # 3) Evaluar
            metrics = self.evaluate(X_train, X_test, y_train, y_test)
            metrics = {k: float(v) for k, v in metrics.items()}
            mlflow.log_metrics(metrics)

            # 4) Residuales (figura) + log
            y_pred = self.model.predict(X_test)
            resid_fig = "reports/figures/residuals_rf.png"
            self._plot_residuals(y_test, y_pred, resid_fig)
            mlflow.log_artifact(resid_fig)

            # 5) Guardar métricas a archivo (para DVC) + log
            metrics_path = "reports/metrics_rf.json"
            import os as _os
            _os.makedirs("reports", exist_ok=True)   # <-- FALTA ESTO EN RF
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(metrics_path)

            # 6) Guardar modelo .pkl con timestamp + log
            saved_path = self.save_model(model_type=model_type, timestamp=timestamp)
            mlflow.log_artifact(saved_path)

            # 7) Subir el modelo en formato MLflow SIN usar logged-models (carpeta temporal)
            from tempfile import TemporaryDirectory
            from pathlib import Path
            with TemporaryDirectory() as tmp:
                local_dir = Path(tmp) / "rf_mlflow_model"
                mlflow.sklearn.save_model(self.model, path=str(local_dir))
                mlflow.log_artifacts(str(local_dir), artifact_path="model")

            print(f"[INFO] Random Forest model saved at: {saved_path}")
            print("[INFO] Random Forest training pipeline complete.\n")

        return metrics

class PipelineModelTrainer(ModelTrainer):
    """
    Versión v2 del entrenador que encapsula la ingeniería de atributos dentro de
    un Pipeline de scikit-learn.
    """

    def __init__(
        self,
        model_params=None,
        training_params=None,
        *,
        feature_config: FeatureConfig | None = None,
        drop_na: bool = False,
        use_date_features: bool = True,
    ):
        super().__init__(model_params=model_params, training_params=training_params)
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
        estimator = RandomForestRegressor(**self.model_params)

        self.model = Pipeline(
            steps=[
                ("feature_engineering", feature_engineering),
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )
