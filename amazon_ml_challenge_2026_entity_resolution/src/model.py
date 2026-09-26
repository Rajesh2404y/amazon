import os
import joblib
import logging
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional

logger = logging.getLogger("entity_resolution")

class BaselineRuleMatcher:
    """Baseline 1 & 2: Rule-based and weighted similarity matching."""
    def __init__(self, mode: str = "exact", threshold: float = 0.85):
        self.mode = mode
        self.threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X[:, 0] is name_exact_match
        # X[:, 1] is name_jaro_winkler
        # X[:, 10] is addr_token_sort_ratio
        if self.mode == "exact":
            # Exact name match
            scores = X[:, 0]
        else:
            # Weighted similarity rule: 0.6 * name_jw + 0.4 * addr_sort
            scores = (0.65 * X[:, 1]) + (0.35 * X[:, 10])
        return scores

class EntityXGBoostMatcher:
    """Production-grade XGBoost matcher with hist tree method for CPU efficiency."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        logger.info(f"Training XGBoost classifier on {X_train.shape[0]:,} pairs...")
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=50
        )
        logger.info("XGBoost model training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
