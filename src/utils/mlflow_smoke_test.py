# src/utils/mlflow_smoke_test.py
from pathlib import Path
import os, mlflow
from src.utils.env import load_env

def run_smoke_test():
    env_vars = load_env()
    Path("reports").mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(env_vars["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(env_vars["EXPERIMENT_NAME"])

    with mlflow.start_run() as run:
        print(f"🚀 Iniciando run: {run.info.run_id}")
        mlflow.set_tags({"stage": "dev", "who": "ale"})
        mlflow.log_params({"probe": "smoke_test"})
        mlflow.log_metrics({"mae": 1.23, "r2": 0.89})
        with open("reports/smoke.txt", "w") as f:
            f.write("Hello MLflow!")
        mlflow.log_artifact("reports/smoke.txt")

        print("✅ Run completado y artifact registrado.")
        print(f"Ver en UI: {env_vars['MLFLOW_TRACKING_URI']}")
        print(f"Experimento: {env_vars['EXPERIMENT_NAME']}")

if __name__ == "__main__":
    run_smoke_test()
