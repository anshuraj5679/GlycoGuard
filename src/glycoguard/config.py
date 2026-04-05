from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class DataConfig:
    resample_rule: str = "5min"
    interpolation_limit: int = 9
    window_size: int = 24
    horizon_steps: int = 6
    hypo_threshold: float = 70.0
    severe_threshold: float = 54.0
    demo_days: int = 14


@dataclass(slots=True)
class ModelConfig:
    strict_mode: bool = True
    n_splits: int = 5
    gap: int = 72
    random_state: int = 42
    prefer_tft: bool = True
    classifier_weight: float = 0.65
    forecast_weight: float = 0.35
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4
    ood_quantile: float = 0.995
    minimum_real_patients: int = 2
    artifact_dir: str = "artifacts/current"


@dataclass(slots=True)
class ApiConfig:
    title: str = "GlycoGuard API"
    version: str = "0.1.0"


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data = DataConfig(**raw.get("data", {}))
    model = ModelConfig(**raw.get("model", {}))
    api = ApiConfig(**raw.get("api", {}))
    return AppConfig(data=data, model=model, api=api)
