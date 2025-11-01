# scripts/test_linear_run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.linear_regression_model.model_trainer import ModelTrainer
from src.utils.env import load_env

CSV_PATH = "data/clean/steel_energy_cleaned.csv"
TARGET = "usage_kwh"  # ya comprobaste que existe y es numérico tras to_numeric

def main():
    # Carga .env (MLFLOW_TRACKING_URI, EXPERIMENT_NAME, etc.)
    load_env()

    # Carga datos
    df = pd.read_csv(CSV_PATH)

    # Asegura target numérico (si viniera como string)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Quedarnos con columnas numéricas para Linear básico
    num_df = df.select_dtypes(include=["number"]).dropna(subset=[TARGET]).copy()

    y = num_df[TARGET]
    X = num_df.drop(columns=[TARGET])

    if X.shape[1] == 0:
        raise ValueError("No quedaron columnas numéricas en X. Revisa el dataset o agrega preprocesamiento.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrena y loguea en MLflow (usa tu instrumentación del trainer)
    trainer = ModelTrainer(use_mlflow=True, tags={"model_type": "linear"})
    metrics = trainer.run(X_train, X_test, y_train, y_test, model_type="linear_regression")
    print("[OK] Métricas:", metrics)

if __name__ == "__main__":
    main()
