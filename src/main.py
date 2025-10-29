# src/main.py
import os
import pandas as pd
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.models.xgboost_model import ModelTrainer


def main():
    # ------------------------------------------------------------
    # STEP 1: Data Loading & Cleaning
    # ------------------------------------------------------------
    raw_path = r"D:\Projects\steel-energy-mlops\data\clean\steel_energy_cleaned_V2.csv"
    processed_path = r"D:\Projects\steel-energy-mlops\data\processed\steel_energy_processed.csv"

    print("=" * 70)
    print("[INFO] STEP 1: Running DataLoader")
    print("=" * 70)

    loader = DataLoader(raw_path, processed_path)
    df_processed = loader.run()

    if os.path.exists(processed_path):
        print(f"[INFO] Processed file found at: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        raise FileNotFoundError("[ERROR] Processed dataset not found after DataLoader step!")

    # ------------------------------------------------------------
    # STEP 2: Feature Engineering
    # ------------------------------------------------------------
    print("=" * 70)
    print("[INFO] STEP 2: Running Feature Engineering")
    print("=" * 70)

    features = [
        "lagging_current_reactive.power_kvarh",
        "leading_current_reactive_power_kvarh",
        "lagging_current_power_factor",
        "leading_current_power_factor",
        "nsm",
        "hour", "dayofweek_num", "month",
        "weekstatus_Weekend",
        "load_type_Maximum_load",
        "load_type_Medium_load"
    ]
    target = "usage_kwh"

    fe = FeatureEngineer(features, target)
    X_train, X_test, y_train, y_test = fe.run(df)

    print(f"[INFO] Feature matrix: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"[INFO] Target: y_train {y_train.shape}, y_test {y_test.shape}")

    # ------------------------------------------------------------
    # STEP 3: Model Training (XGBoost)
    # ------------------------------------------------------------
    print("=" * 70)
    print("[INFO] STEP 3: Training XGBoost Model")
    print("=" * 70)

    trainer = ModelTrainer()
    metrics = trainer.run(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------
    # STEP 4: Cross-Validation (Optional)
    # ------------------------------------------------------------
    print("=" * 70)
    print("[INFO] STEP 4: Cross-Validation")
    print("=" * 70)
    trainer.cross_validate(X_train, y_train)

    # ------------------------------------------------------------
    # STEP 5: Pipeline Summary
    # ------------------------------------------------------------
    print("=" * 70)
    print("[INFO] STEP 5: Pipeline Summary")
    print("=" * 70)
    print(f"Processed dataset: {df.shape}")
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test  set: X={X_test.shape}, y={y_test.shape}")
    print(f"Model Metrics: {metrics}")
    print("\n[INFO] ✅ Full pipeline executed successfully!")


if __name__ == "__main__":
    main()

