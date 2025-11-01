import os
import joblib
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from .config import MODEL_CONFIG, TRAINING_CONFIG

# MLflow + utilidades
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from src.utils.env import load_env  # para cargar el .env


class ModelTrainer:
    """
    Trains, evaluates, and validates a Random Forest regression model.
    """

    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or MODEL_CONFIG
        self.training_params = training_params or TRAINING_CONFIG
        self.model = RandomForestRegressor(**self.model_params)

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
        mlflow.set_experiment(env_vars["EXPERIMENT_NAME"])
        self._ensure_output_dirs()

        print("[INFO] Starting Random Forest training pipeline...")

        # TODO IMPORTANTE: todo lo de entrenamiento y logging va DENTRO del with
        with mlflow.start_run():
            # Registrar parámetros generales
            mlflow.set_tags({"model_family": model_type, "stage": "dev"})
            mlflow.log_params({f"model__{k}": v for k, v in self.model.get_params().items()})
            mlflow.log_params({f"train__{k}": v for k, v in self.training_params.items()})
            mlflow.log_params({
                "n_train": int(getattr(X_train, "shape", [len(X_train)])[0]),
                "n_test": int(getattr(X_test, "shape", [len(X_test)])[0])
            })

            # 1) Entrenar
            self.train(X_train, y_train)

            # 2) Evaluar
            metrics = self.evaluate(X_train, X_test, y_train, y_test)
            # (asegura floats nativos)
            metrics = {k: float(v) for k, v in metrics.items()}
            mlflow.log_metrics(metrics)

            # 3) Residuales (figura) + log
            y_pred = self.model.predict(X_test)
            resid_fig = "reports/figures/residuals_rf.png"
            self._plot_residuals(y_test, y_pred, resid_fig)
            mlflow.log_artifact(resid_fig)

            # 4) Guardar métricas a archivo + log
            metrics_path = "reports/metrics.json"
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(metrics_path)

           # 5) Guardar modelo .pkl con timestamp + log
            saved_path = self.save_model(model_type=model_type, timestamp=timestamp)
            mlflow.log_artifact(saved_path)

            # 6) Subir el modelo en formato MLflow SIN usar el endpoint de logged-models
            from tempfile import TemporaryDirectory
            from pathlib import Path
            tmp_ok = False
            with TemporaryDirectory() as tmp:
                local_dir = Path(tmp) / "rf_mlflow_model"
                # guarda a disco en formato MLflow
                mlflow.sklearn.save_model(self.model, path=str(local_dir))
                # súbelo como artifacts normales bajo 'model/'
                mlflow.log_artifacts(str(local_dir), artifact_path="model")
                tmp_ok = True
            print(f"[INFO] Model directory logged as artifacts under 'model/' (ok={tmp_ok})")

            print(f"[INFO] Random Forest model saved at: {saved_path}")
            print(f"[INFO] Random Forest training pipeline complete.\n")

        return metrics
