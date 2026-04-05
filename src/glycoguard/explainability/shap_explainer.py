from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


FEATURE_TEXT = {
    "roc_15": "Glucose is dropping rapidly over the last 15 minutes.",
    "roc_30": "The 30-minute glucose trend is strongly downward.",
    "min_2h": "Recent glucose lows suggest limited safety margin.",
    "carbs_1h": "Recent carbohydrate intake is buffering risk.",
    "carbs_2h": "Carbohydrate coverage over the last 2 hours is protective.",
    "insulin_on_board": "Active insulin is still lowering glucose.",
    "activity": "Current activity is increasing glucose utilization.",
    "activity_6h": "This looks like a delayed post-exercise risk window.",
    "sleep_flag": "This is a nocturnal period with reduced awareness.",
    "lbgi_2h": "Low Blood Glucose Index is elevated.",
    "glucose_deficit_2h": "Recent time below target increases near-term risk.",
    "time_since_last_meal_min": "A long fasting interval raises risk.",
}


class HypoExplainer:
    def __init__(self, model: Any, X_background: pd.DataFrame) -> None:
        self.model = getattr(model, "estimator", model)
        self.feature_names = X_background.columns.tolist()
        if shap is None:
            raise RuntimeError("SHAP is required in strict mode. Install shap before serving explanations.")
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"TreeExplainer initialisation failed in strict mode: {type(exc).__name__}: {exc}") from exc
        self.backend = "shap"

    def _message(self, feature: str, contribution: float, value: float) -> str:
        prefix = "Increases risk" if contribution >= 0 else "Reduces risk"
        default = f"{feature} contributes {contribution:.2f}."
        if feature in FEATURE_TEXT:
            return f"{prefix}: {FEATURE_TEXT[feature]}"
        return f"{prefix}: {default}"

    def explain(self, X_sample: pd.DataFrame) -> dict[str, object]:
        row = X_sample.iloc[0]
        raw_values = self.explainer.shap_values(X_sample)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[-1])[0]
        else:
            values = np.asarray(raw_values)
            if values.ndim == 2:
                values = values[0]
            elif values.ndim == 3:
                values = values[0, :, -1]
        contributions = {feature: float(value) for feature, value in zip(self.feature_names, values)}
        expected = getattr(self.explainer, "expected_value", 0.0)
        expected = np.asarray(expected)
        base_value = float(expected.ravel()[-1]) if expected.size else 0.0

        sorted_items = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_factors = [
            {
                "feature": feature,
                "contribution": float(value),
                "message": self._message(feature, value, float(row.get(feature, 0.0))),
            }
            for feature, value in sorted_items[:5]
        ]
        explanation = " ".join(factor["message"] for factor in top_factors[:3])
        return {
            "shap_values": contributions,
            "top_factors": top_factors,
            "explanation": explanation,
            "backend": self.backend,
            "waterfall": {
                "base_value": base_value,
                "feature_names": self.feature_names,
                "feature_values": {feature: float(row.get(feature, 0.0)) for feature in self.feature_names},
                "shap_values": contributions,
                "backend": self.backend,
            },
        }
