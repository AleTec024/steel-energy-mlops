import os
import joblib
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from .config import MODEL_CONFIG, TRAINING_CONFIG

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

    def save_model(self, output_dir="models/current", model_type="random_forest", timestamp=None):
        """
        Save model artifact under unique versioned filename
        and keep a 'current' copy for latest reference.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 🔹 Create timestamp for unique version naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 🔹 Define both versioned and latest file paths
        versioned_dir = f"models/{model_type}/artifacts"
        os.makedirs(versioned_dir, exist_ok=True)
        versioned_model_path = os.path.join(versioned_dir, f"model_{timestamp}.pkl")
        latest_model_path = os.path.join(output_dir, "model.pkl")

        # 🔹 Save model depending on type
        if model_type == "xgboost":
            self.model.save_model(versioned_model_path)
        else:
            joblib.dump(self.model, versioned_model_path)

        # 🔹 Also copy to 'latest'
        joblib.dump(self.model, latest_model_path)

        print(f"[INFO] Saved versioned model to: {versioned_model_path}")
        print(f"[INFO] Updated latest model: {latest_model_path}")

        return versioned_model_path
    
    def run(self, X_train, X_test, y_train, y_test, model_type="random_forest", timestamp=None):
        """
        Full training + evaluation pipeline for Random Forest.
        Saves model with timestamped filename and logs metrics.
        """
        print("[INFO] Starting Random Forest training pipeline...")

        # 1. Train model
        self.train(X_train, y_train)

        # 2. Evaluate performance
        metrics = self.evaluate(X_train, X_test, y_train, y_test)

        # 3. Save model (versioned + current)
        saved_path = self.save_model(model_type=model_type, timestamp=timestamp)

        print(f"[INFO] Random Forest model saved at: {saved_path}")
        print(f"[INFO] Random Forest training pipeline complete.\n")

        return metrics

    
