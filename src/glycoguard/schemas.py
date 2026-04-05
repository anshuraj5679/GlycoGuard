from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CGMInput(BaseModel):
    glucose_readings: List[float] = Field(..., description="Last 24 CGM readings.", min_length=24, max_length=24)
    patient_id: Optional[str] = None
    carbs_last_hour: float = 0.0
    carbs_last_2h: Optional[float] = None
    insulin_on_board: float = 0.0
    activity_level: float = 0.0
    sleep_flag: int = 0
    stress_score: float = 0.0
    hour_of_day: Optional[int] = None
    timestamp: Optional[datetime] = None

    @field_validator("glucose_readings")
    @classmethod
    def validate_glucose_readings(cls, value: List[float]) -> List[float]:
        if any((reading < 40.0 or reading > 400.0) for reading in value):
            raise ValueError("Glucose readings must stay within 40 to 400 mg/dL.")
        return value

    @field_validator("carbs_last_hour")
    @classmethod
    def validate_carbs_last_hour(cls, value: float) -> float:
        if not 0.0 <= value <= 150.0:
            raise ValueError("carbs_last_hour must be between 0 and 150 grams.")
        return value

    @field_validator("carbs_last_2h")
    @classmethod
    def validate_carbs_last_2h(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0.0 <= value <= 250.0:
            raise ValueError("carbs_last_2h must be between 0 and 250 grams.")
        return value

    @field_validator("insulin_on_board")
    @classmethod
    def validate_insulin_on_board(cls, value: float) -> float:
        if not 0.0 <= value <= 25.0:
            raise ValueError("insulin_on_board must be between 0 and 25 units.")
        return value

    @field_validator("activity_level", "stress_score")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Activity and stress inputs must be normalised to the 0-1 range.")
        return value

    @field_validator("sleep_flag")
    @classmethod
    def validate_sleep_flag(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("sleep_flag must be 0 or 1.")
        return value


class PatientProfile(BaseModel):
    patient_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    diabetes_type: Optional[str] = None
    insulin_therapy: Optional[str] = None
    target_range_low: float = 70.0
    target_range_high: float = 180.0
    weight_kg: Optional[float] = None


class PatientProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    diabetes_type: Optional[str] = None
    insulin_therapy: Optional[str] = None
    target_range_low: Optional[float] = None
    target_range_high: Optional[float] = None
    weight_kg: Optional[float] = None


class MealLogRequest(BaseModel):
    patient_id: str
    carb_grams: float
    timestamp: Optional[datetime] = None
    description: Optional[str] = None

    @field_validator("carb_grams")
    @classmethod
    def validate_carb_grams(cls, value: float) -> float:
        if not 0.0 <= value <= 150.0:
            raise ValueError("carb_grams must be between 0 and 150 grams.")
        return value


class InsulinLogRequest(BaseModel):
    patient_id: str
    insulin_units: float
    timestamp: Optional[datetime] = None
    insulin_type: str = "bolus"

    @field_validator("insulin_units")
    @classmethod
    def validate_insulin_units(cls, value: float) -> float:
        if not 0.0 <= value <= 25.0:
            raise ValueError("insulin_units must be between 0 and 25 units.")
        return value


class OhioIngestRequest(BaseModel):
    root_dir: str
    split: str = "test"
    patient_ids: Optional[List[str]] = None
    prefix: str = "ohio"
    retrain: bool = False
    persist: bool = False


class ReplayStartRequest(BaseModel):
    patient_id: Optional[str] = None
    start_cursor: Optional[int] = None


class WatchPayloadResponse(BaseModel):
    patient_id: str
    glucose: float
    roc_15: Optional[float] = None
    trend: str
    risk: str
    reason: str
    buzz: bool
    forecast_30min: Optional[float] = None
    forecast_warning: str = ""
    hypo_probability: Optional[float] = None
    top_reason: str = ""
    watch_status: str = ""
    updated_at: str
    status: str = "live"
    session_id: Optional[str] = None


class ReplayStepResponse(BaseModel):
    session_id: str
    patient_id: str
    cursor: int
    total_steps: int
    progress: float
    status: str
    timestamp: str
    watch: WatchPayloadResponse
    actual_glucose_30min: float
    actual_hypo: bool


class ExplanationFactor(BaseModel):
    feature: str
    contribution: float
    message: str


class WaterfallPayload(BaseModel):
    base_value: float
    feature_names: List[str]
    feature_values: Dict[str, float]
    shap_values: Dict[str, float]
    backend: str


class PredictionResponse(BaseModel):
    patient_id: str
    status: str = "ok"
    current_glucose: Optional[float] = None
    roc_15: Optional[float] = None
    hypo_probability: Optional[float] = None
    classifier_probability: Optional[float] = None
    forecast_probability: Optional[float] = None
    risk_level: Optional[str] = None
    prob_risk: Optional[str] = None
    forecast_risk: Optional[str] = None
    predicted_glucose_30min: Optional[float] = None
    forecast_trace: List[float] = Field(default_factory=list)
    forecast_lower: List[float] = Field(default_factory=list)
    forecast_upper: List[float] = Field(default_factory=list)
    forecast_notice: Optional[str] = None
    alert_required: bool
    watch_buzz: bool = False
    forecast_warning: str = ""
    top_reason: str = ""
    watch_status: str = ""
    model_backend: str
    forecast_backend: str
    confidence: Optional[float] = None
    abstention_reason: Optional[str] = None


class ExplanationResponse(PredictionResponse):
    explanation: str = ""
    top_factors: List[ExplanationFactor] = Field(default_factory=list)
    shap_values: Dict[str, float] = Field(default_factory=dict)
    waterfall: Optional[WaterfallPayload] = None


class HealthResponse(BaseModel):
    status: str
    api_version: str
    ready: bool
    model_backend: Optional[str] = None
    forecast_backend: Optional[str] = None
    default_patient_id: Optional[str] = None
    strict_mode: bool
    real_patient_count: int = 0


class TrainRequest(BaseModel):
    patient_ids: Optional[List[str]] = None
    persist: bool = False


class BundleIngestRequest(BaseModel):
    bundle_dir: str
    patient_id: Optional[str] = None
    retrain: bool = False
    persist: bool = False


class ArtifactRequest(BaseModel):
    directory: Optional[str] = None


class AuditRequest(BaseModel):
    patient_id: Optional[str] = None
    payload: Optional[CGMInput] = None
    current_glucose: Optional[float] = None
    use_gemini: bool = False
    model: Optional[str] = None
    timeout_seconds: float = 30.0


class OhioBenchmarkRequest(BaseModel):
    root_dir: str
    persist: bool = False
    max_forecast_points: int = 500
    train_patient_ids: Optional[List[str]] = None
    test_patient_ids: Optional[List[str]] = None


class FederatedRunRequest(BaseModel):
    patient_ids: Optional[List[str]] = None
    rounds: int = 3
    min_clients: int = 2
