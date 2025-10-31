MODEL_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,   # let RF grow; tune later
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "random_state": 42,
    "n_jobs": -1
}

TRAINING_CONFIG = {
    "cv_folds": 5,
    "test_size": 0.2,
}
