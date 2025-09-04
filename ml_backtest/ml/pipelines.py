"""
ML pipelines for classification-based strategies.

Provides calibrated probability models:
- `log_reg`: Standardised Logistic Regression wrapped with sigmoid calibration.
- `random_forest`: RandomForest with isotonic calibration.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def build_model(name: str = "log_reg"):
    """Create a calibrated classifier by name.

    Args:
        name (str): "log_reg" or "random_forest".

    Returns:
        A scikit-learn estimator supporting `fit` and `predict_proba`.
    """
    if name == "log_reg":
        base = Pipeline([("scaler", StandardScaler(with_mean=True)),
                         ("clf", LogisticRegression(max_iter=1000))])
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    elif name == "random_forest":
        base = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model

def fit_predict_proba(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    """Fit model on (X_train, y_train) and return class-1 probabilities for X_test."""
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1]
