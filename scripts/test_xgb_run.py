# scripts/test_xgb_run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.xgboost_model.model_trainer import ModelTrainer
from src.utils.env import load_env

CSV_PATH = "data/clean/steel_energy_cleaned.csv"
TARGET = "usage_kwh"

def main():
    load_env()
    df = pd.read_csv(CSV_PATH)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    num_df = df.select_dtypes(include=["number"]).dropna(subset=[TARGET]).copy()

    y = num_df[TARGET]
    X = num_df.drop(columns=[TARGET])

    if X.shape[1] == 0:
        raise ValueError("No quedaron columnas numéricas en X. Revisa el dataset o agrega preprocesamiento.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainer = ModelTrainer()  # usa tus MODEL_CONFIG/TRAINING_CONFIG
    metrics = trainer.run(X_train, X_test, y_train, y_test, model_type="xgboost")
    print("[OK] Métricas:", metrics)

if __name__ == "__main__":
    main()
