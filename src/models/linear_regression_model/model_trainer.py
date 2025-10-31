import os
import joblib
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from .config import MODEL_CONFIG, TRAINING_CONFIG

class ModelTrainer:
    """
    Trains, evaluates, and validates a Linear Regression model.
    """

    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or MODEL_CONFIG
        self.training_params = training_params or TRAINING_CONFIG
        self.model = LinearRegression(**self.model_params)

    def train(self, X_train, y_train):
        print("[INFO] Training Linear Regression model...")
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

    def save_model(self, model_type="linear_regression", timestamp=None):
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


    def run(self, X_train, X_test, y_train, y_test, model_type="linear_regression", timestamp=None):
        """
        Full training + evaluation pipeline for Linear Regression.
        Saves model with timestamped filename and logs metrics.
        """
        print("[INFO] Starting Linear Regression training pipeline...")

        # 1. Train model
        self.train(X_train, y_train)

        # 2. Evaluate performance
        metrics = self.evaluate(X_train, X_test, y_train, y_test)

        # 3. Save model (versioned + current)
        saved_path = self.save_model(model_type=model_type, timestamp=timestamp)

        print(f"[INFO] Linear Regression model saved at: {saved_path}")
        print(f"[INFO] Linear Regression training pipeline complete.\n")

        return metrics

