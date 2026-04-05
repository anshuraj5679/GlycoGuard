from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import pandas as pd

from glycoguard.ingestion.loader import align_context, prepare_cgm_frame


@dataclass(slots=True)
class OhioSplitPaths:
    train_dir: Path
    test_dir: Path


def _local_name(tag: str) -> str:
    return tag.split("}")[-1].lower()


def discover_ohio_split_dirs(root_dir: str | Path) -> OhioSplitPaths:
    root = Path(root_dir)
    candidates = {
        "train": ["OhioT1DM-training", "ohio-training", "train", "training"],
        "test": ["OhioT1DM-testing", "ohio-testing", "test", "testing"],
    }

    def resolve(which: str) -> Path:
        for name in candidates[which]:
            candidate = root / name
            if candidate.exists() and candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"Could not find the OhioT1DM {which} directory under {root}")

    return OhioSplitPaths(train_dir=resolve("train"), test_dir=resolve("test"))


def _to_timestamp(attrs: dict[str, str]) -> pd.Timestamp | None:
    for key in ("ts", "timestamp", "time", "date", "start_time", "start"):
        if key in attrs:
            ts = pd.to_datetime(attrs[key], errors="coerce")
            if not pd.isna(ts):
                return pd.Timestamp(ts)
    return None


def _to_float(attrs: dict[str, str], candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        if key in attrs:
            try:
                return float(attrs[key])
            except (TypeError, ValueError):
                continue
    return None


def _append_interval_samples(
    rows: list[dict[str, float | pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    key: str,
    value: float,
) -> None:
    if end <= start:
        rows.append({"timestamp": start, key: value})
        return
    for ts in pd.date_range(start=start, end=end, freq="5min"):
        rows.append({"timestamp": ts, key: value})


def parse_ohio_xml(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = ET.parse(path).getroot()
    cgm_rows: list[dict[str, object]] = []
    meal_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    insulin_rows: list[dict[str, object]] = []

    for elem in root.iter():
        tag = _local_name(elem.tag)
        attrs = {_local_name(key): value for key, value in elem.attrib.items()}
        timestamp = _to_timestamp(attrs)
        if timestamp is None:
            continue

        glucose_value = _to_float(attrs, ("value", "glucose", "cgm", "mgdl"))
        carb_value = _to_float(attrs, ("carbs", "carb", "carbohydrates", "grams", "value"))
        bolus_value = _to_float(attrs, ("dose", "amount", "value", "units", "bolus"))
        basal_rate = _to_float(attrs, ("rate", "value", "basal"))
        intensity = _to_float(attrs, ("intensity", "value", "level"))
        duration = _to_float(attrs, ("duration", "minutes", "mins", "duration_minutes"))

        if "glucose" in tag and glucose_value is not None:
            cgm_rows.append({"timestamp": timestamp, "glucose": glucose_value})
            continue

        if ("meal" in tag or "carb" in tag) and carb_value is not None:
            meal_rows.append({"timestamp": timestamp, "carb_grams": carb_value})
            continue

        if "bolus" in tag and bolus_value is not None:
            insulin_rows.append({"timestamp": timestamp, "insulin_units": bolus_value})
            continue

        if "basal" in tag and basal_rate is not None:
            insulin_rows.append({"timestamp": timestamp, "basal_rate": basal_rate})
            continue

        if any(keyword in tag for keyword in ("exercise", "activity", "work")):
            value = 0.5 if intensity is None else max(0.1, intensity)
            if duration and duration > 5:
                end = timestamp + pd.Timedelta(minutes=float(duration))
                _append_interval_samples(activity_rows, timestamp, end, "activity", float(value))
            else:
                activity_rows.append({"timestamp": timestamp, "activity": float(value)})
            continue

        if "sleep" in tag:
            end_time = pd.to_datetime(attrs.get("end") or attrs.get("end_time"), errors="coerce")
            if pd.notna(end_time):
                _append_interval_samples(activity_rows, timestamp, pd.Timestamp(end_time), "sleep_flag", 1.0)
            else:
                activity_rows.append({"timestamp": timestamp, "sleep_flag": 1.0})
            continue

        if "stress" in tag:
            stress_value = 0.6 if intensity is None else max(0.1, intensity)
            activity_rows.append({"timestamp": timestamp, "stress_score": float(stress_value)})

    cgm = pd.DataFrame(cgm_rows)
    meals = pd.DataFrame(meal_rows)
    activity = pd.DataFrame(activity_rows)
    insulin = pd.DataFrame(insulin_rows)
    return cgm, meals, activity, insulin


def load_ohio_patient_xml(
    path: str | Path,
    patient_id: str | None = None,
    resample_rule: str = "5min",
    interpolation_limit: int = 9,
) -> tuple[str, pd.DataFrame]:
    cgm_raw, meals, activity, insulin = parse_ohio_xml(path)
    if cgm_raw.empty:
        raise ValueError(f"No glucose rows were found in OhioT1DM XML: {path}")

    cgm = prepare_cgm_frame(
        cgm_raw,
        resample_rule=resample_rule,
        interpolation_limit=interpolation_limit,
    )
    aligned = align_context(
        cgm,
        meals_df=meals if not meals.empty else None,
        activity_df=activity if not activity.empty else None,
        insulin_df=insulin if not insulin.empty else None,
        resample_rule=resample_rule,
    )
    resolved_id = patient_id or Path(path).stem
    return resolved_id, aligned


def load_ohio_split(
    root_dir: str | Path,
    split: str,
    resample_rule: str = "5min",
    interpolation_limit: int = 9,
    patient_ids: Optional[set[str]] = None,
) -> dict[str, pd.DataFrame]:
    split_dirs = discover_ohio_split_dirs(root_dir)
    target_dir = split_dirs.train_dir if split.lower() == "train" else split_dirs.test_dir
    payload: dict[str, pd.DataFrame] = {}
    for xml_path in sorted(target_dir.glob("*.xml")):
        patient_id = xml_path.stem
        if patient_ids is not None and patient_id not in patient_ids:
            continue
        resolved_id, frame = load_ohio_patient_xml(
            xml_path,
            patient_id=patient_id,
            resample_rule=resample_rule,
            interpolation_limit=interpolation_limit,
        )
        payload[resolved_id] = frame
    if not payload:
        raise ValueError(f"No OhioT1DM XML files found in {target_dir}")
    return payload

