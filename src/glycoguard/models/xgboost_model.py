from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from glycoguard.features.engineer import FEATURE_COLUMNS

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


@dataclass(slots=True)
class TabularModelBundle:
    estimator: Any
    backend: str
    feature_names: list[str]
    metrics: dict[str, float]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.loc[:, self.feature_names]
        return self.estimator.predict_proba(frame)[:, 1]


def _compute_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    preds = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = recall_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probabilities) if y_true.nunique() > 1 else 0.5
    return {
        "auc_roc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "alert_rate": float(preds.mean()),
        "prevalence": float(np.mean(y_true)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _build_estimator(scale_pos_weight: float, random_state: int) -> tuple[Any, str]:
    if xgb is None:
        raise RuntimeError("xgboost is required in strict mode. Install the package before training or serving.")
    estimator = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=4,
    )
    return estimator, "xgboost"


def train_xgboost(
    feature_df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 72,
    random_state: int = 42,
) -> TabularModelBundle:
    X = feature_df.loc[:, list(FEATURE_COLUMNS)]
    y = feature_df["hypo_label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training data must contain both hypo and non-hypo samples.")

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    metrics: list[dict[str, float]] = []

    for train_idx, val_idx in splitter.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        estimator, _ = _build_estimator(neg / pos, random_state)

        if xgb is not None and estimator.__class__.__module__.startswith("xgboost"):
            fit_kwargs = {"verbose": False}
            # Skip AUC-based eval callbacks on one-class validation folds; they only add noise.
            if y_val.nunique() > 1:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            estimator.fit(X_train, y_train, **fit_kwargs)
        else:
            estimator.fit(X_train, y_train)

        probabilities = estimator.predict_proba(X_val)[:, 1]
        metrics.append(_compute_metrics(y_val, probabilities))

    full_pos = max(1, int((y == 1).sum()))
    full_neg = max(1, int((y == 0).sum()))
    final_estimator, backend = _build_estimator(full_neg / full_pos, random_state)
    final_estimator.fit(X, y)

    mean_metrics = {
        key: float(np.mean([fold[key] for fold in metrics]))
        for key in metrics[0]
    }
    return TabularModelBundle(
        estimator=final_estimator,
        backend=backend,
        feature_names=list(FEATURE_COLUMNS),
        metrics=mean_metrics,
    )
