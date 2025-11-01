# scripts/test_rf_run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.random_forest_model.model_trainer import ModelTrainer
from src.utils.env import load_env

CSV_PATH = "data/clean/steel_energy_cleaned.csv" 
TARGET_CANDIDATES = [
    "usage_kwh", "Usage_kwh", "USAGE_KWH", "usage_kWh",
    "energy_kwh", "energy_usage_kwh", "target", "TARGET"
]

def pick_target(df):
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        for c in df.columns:
            if c.lower() == lc:
                return c
    for c in df.columns:
        cl = c.lower()
        if "usage" in cl and "kwh" in cl:
            return c
    return None

def main():
    load_env()
    df = pd.read_csv(CSV_PATH)

    print("[Cols]", list(df.columns))
    print("[Dtypes]\n", df.dtypes.head(30))
    print(df.head(3))

    target = pick_target(df)
    if not target:
        raise ValueError(f"No encontré el target por alias/heurística. Columnas: {list(df.columns)}")

    print(f"[INFO] Target detectado: {target}")
    df[target] = pd.to_numeric(df[target], errors="coerce")

    num_df = df.select_dtypes(include=["number"]).copy()
    if target not in num_df.columns:
        num_df[target] = df[target]

    before = len(num_df)
    num_df = num_df.dropna(subset=[target])
    after = len(num_df)
    if after < before:
        print(f"[WARN] Se removieron {before - after} filas por target NaN.")

    X = num_df.drop(columns=[target])
    y = num_df[target]
    if X.shape[1] == 0:
        raise ValueError("No quedaron columnas numéricas en X. Revisa el dataset o agrega preprocesamiento.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainer = ModelTrainer()
    metrics = trainer.run(X_train, X_test, y_train, y_test, model_type="random_forest")
    print("[OK] Métricas:", metrics)

if __name__ == "__main__":
    main()
