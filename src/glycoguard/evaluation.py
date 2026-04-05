from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score


@dataclass(slots=True)
class BinaryMetricBundle:
    auc_roc: float
    sensitivity: float
    specificity: float
    f1: float
    alert_rate: float
    prevalence: float
    tp: float
    fp: float
    fn: float
    tn: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc_roc": self.auc_roc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "f1": self.f1,
            "alert_rate": self.alert_rate,
            "prevalence": self.prevalence,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def compute_binary_metrics(y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    truth = np.asarray(list(y_true), dtype=int)
    probs = np.asarray(list(probabilities), dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = recall_score(truth, preds, zero_division=0)
    auc = roc_auc_score(truth, probs) if len(np.unique(truth)) > 1 else 0.5
    bundle = BinaryMetricBundle(
        auc_roc=float(auc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1=float(f1_score(truth, preds, zero_division=0)),
        alert_rate=float(preds.mean()) if len(preds) else 0.0,
        prevalence=float(truth.mean()) if len(truth) else 0.0,
        tp=float(tp),
        fp=float(fp),
        fn=float(fn),
        tn=float(tn),
    )
    return bundle.as_dict()


def compute_lead_time(
    glucose: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.4,
    hypo_threshold: float = 70.0,
    lookback_minutes: int = 60,
) -> dict[str, float | int | list[float]]:
    series = glucose.astype(float).sort_index()
    probs = pd.Series(probabilities, index=series.index).sort_index()
    onset_mask = (series < hypo_threshold) & (series.shift(1, fill_value=series.iloc[0]) >= hypo_threshold)
    event_times = series.index[onset_mask]
    leads: list[float] = []

    for onset in event_times:
        window_start = onset - pd.Timedelta(minutes=lookback_minutes)
        alerts = probs.loc[(probs.index >= window_start) & (probs.index < onset) & (probs >= threshold)]
        if not alerts.empty:
            first_alert = alerts.index[0]
            leads.append(float((onset - first_alert).total_seconds() / 60.0))

    if not event_times.empty:
        coverage = len(leads) / len(event_times)
    else:
        coverage = 0.0

    return {
        "num_events": int(len(event_times)),
        "num_events_covered": int(len(leads)),
        "coverage": float(coverage),
        "mean_minutes": float(np.mean(leads)) if leads else 0.0,
        "median_minutes": float(np.median(leads)) if leads else 0.0,
        "max_minutes": float(np.max(leads)) if leads else 0.0,
        "lead_times": [float(value) for value in leads],
    }


def _clarke_zone(reference: float, estimate: float) -> str:
    ref = float(reference)
    est = float(estimate)

    if ref < 70.0 and est < 70.0:
        return "A"
    if ref > 0 and abs(est - ref) / ref <= 0.20:
        return "A"
    if (ref >= 180.0 and est <= 70.0) or (ref <= 70.0 and est >= 180.0):
        return "E"
    if (
        (70.0 <= ref <= 180.0 and (est > 240.0 or est < 70.0))
        or (70.0 <= est <= 180.0 and (ref > 240.0 or ref < 70.0))
    ):
        return "D"
    if (70.0 <= ref <= 290.0 and est >= ref + 110.0) or (130.0 <= ref <= 180.0 and est <= (7.0 / 5.0) * ref - 182.0):
        return "C"
    return "B"


def compute_clarke_grid(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, object]:
    ref = np.asarray(list(actual), dtype=float)
    est = np.asarray(list(predicted), dtype=float)
    zones = [_clarke_zone(r, e) for r, e in zip(ref, est)]
    counts = {zone: zones.count(zone) for zone in ("A", "B", "C", "D", "E")}
    total = max(1, len(zones))
    percentages = {zone: counts[zone] / total for zone in counts}
    return {
        "counts": counts,
        "percentages": percentages,
        "zone_ab": percentages["A"] + percentages["B"],
    }

