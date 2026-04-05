from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(slots=True)
class OODDetector:
    feature_names: list[str]
    location: np.ndarray
    precision: np.ndarray
    threshold: float

    def score(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.loc[:, self.feature_names].to_numpy(dtype=float)
        centered = frame - self.location
        distances = np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        return np.sqrt(np.clip(distances, a_min=0.0, a_max=None))

    def classify(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        distances = self.score(X)
        accepted = distances <= self.threshold
        return accepted, distances


def fit_ood_detector(feature_df: pd.DataFrame, quantile: float = 0.995) -> OODDetector:
    if feature_df.empty:
        raise ValueError("Cannot fit OOD detector on an empty feature frame.")

    frame = feature_df.copy().replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("No finite rows are available for OOD fitting.")

    covariance = LedoitWolf().fit(frame.to_numpy(dtype=float))
    centered = frame.to_numpy(dtype=float) - covariance.location_
    distances = np.sqrt(
        np.clip(
            np.einsum("ij,jk,ik->i", centered, covariance.precision_, centered),
            a_min=0.0,
            a_max=None,
        )
    )
    threshold = float(np.quantile(distances, quantile))
    return OODDetector(
        feature_names=frame.columns.tolist(),
        location=np.asarray(covariance.location_, dtype=float),
        precision=np.asarray(covariance.precision_, dtype=float),
        threshold=threshold,
    )
