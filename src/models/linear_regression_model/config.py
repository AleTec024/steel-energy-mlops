MODEL_CONFIG = {
    # sklearn LinearRegression has few knobs
    "fit_intercept": True,
    "n_jobs": None  # use all cores: set to None in recent sklearn
}

TRAINING_CONFIG = {
    "cv_folds": 5,
    "test_size": 0.2,
}
