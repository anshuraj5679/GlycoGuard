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
    for key in ("ts", "timestamp", "time", "date", "start_time", "start", "ts_begin", "tbegin"):
        if key in attrs:
            ts = pd.to_datetime(attrs[key], errors="coerce", dayfirst=True)
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


def _to_end_timestamp(attrs: dict[str, str]) -> pd.Timestamp | None:
    for key in ("end", "end_time", "ts_end", "tend"):
        if key in attrs:
            ts = pd.to_datetime(attrs[key], errors="coerce", dayfirst=True)
            if not pd.isna(ts):
                return pd.Timestamp(ts)
    return None


def parse_ohio_xml(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = ET.parse(path).getroot()
    cgm_rows: list[dict[str, object]] = []
    meal_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    insulin_rows: list[dict[str, object]] = []

    for section in root:
        section_tag = _local_name(section.tag)
        section_attrs = {_local_name(key): value for key, value in section.attrib.items()}
        if len(section) == 0:
            timestamp = _to_timestamp(section_attrs)
            if section_tag == "glucose_level":
                glucose_value = _to_float(section_attrs, ("value", "glucose", "cgm", "mgdl"))
                if timestamp is not None and glucose_value is not None:
                    cgm_rows.append({"timestamp": timestamp, "glucose": glucose_value})
                continue
            if section_tag == "meal":
                carb_value = _to_float(section_attrs, ("carbs", "carb", "carbohydrates", "grams", "value"))
                if timestamp is not None and carb_value is not None:
                    meal_rows.append({"timestamp": timestamp, "carb_grams": carb_value})
                continue
            if section_tag == "bolus":
                bolus_value = _to_float(section_attrs, ("dose", "amount", "value", "units", "bolus"))
                if timestamp is not None and bolus_value is not None:
                    insulin_rows.append({"timestamp": timestamp, "insulin_units": bolus_value})
                continue
            if section_tag in {"basal", "temp_basal"}:
                basal_rate = _to_float(section_attrs, ("rate", "value", "basal"))
                if timestamp is not None and basal_rate is not None:
                    insulin_rows.append({"timestamp": timestamp, "basal_rate": basal_rate})
                continue
            if section_tag in {"exercise", "work"}:
                if timestamp is None:
                    continue
                intensity = _to_float(section_attrs, ("intensity", "value", "level"))
                duration = _to_float(section_attrs, ("duration", "minutes", "mins", "duration_minutes"))
                value = 0.5 if intensity is None else max(0.1, intensity / 10.0 if intensity > 1.0 else intensity)
                if duration and duration > 5:
                    end = timestamp + pd.Timedelta(minutes=float(duration))
                    _append_interval_samples(activity_rows, timestamp, end, "activity", float(value))
                else:
                    end = _to_end_timestamp(section_attrs)
                    if end is not None and end > timestamp:
                        _append_interval_samples(activity_rows, timestamp, end, "activity", float(value))
                    else:
                        activity_rows.append({"timestamp": timestamp, "activity": float(value)})
                continue
            if section_tag in {"sleep", "basis_sleep"}:
                if timestamp is None:
                    continue
                end = _to_end_timestamp(section_attrs)
                if end is not None and end > timestamp:
                    _append_interval_samples(activity_rows, timestamp, end, "sleep_flag", 1.0)
                else:
                    activity_rows.append({"timestamp": timestamp, "sleep_flag": 1.0})
                continue
            if section_tag in {"stress", "stressors"} and timestamp is not None:
                activity_rows.append({"timestamp": timestamp, "stress_score": _to_float(section_attrs, ("value",)) or 0.7})
                continue

        if section_tag == "glucose_level":
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                glucose_value = _to_float(attrs, ("value", "glucose", "cgm", "mgdl"))
                if timestamp is not None and glucose_value is not None:
                    cgm_rows.append({"timestamp": timestamp, "glucose": glucose_value})
            continue

        if section_tag == "meal":
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                carb_value = _to_float(attrs, ("carbs", "carb", "carbohydrates", "grams", "value"))
                if timestamp is not None and carb_value is not None:
                    meal_rows.append({"timestamp": timestamp, "carb_grams": carb_value})
            continue

        if section_tag == "bolus":
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                bolus_value = _to_float(attrs, ("dose", "amount", "value", "units", "bolus"))
                if timestamp is not None and bolus_value is not None:
                    insulin_rows.append({"timestamp": timestamp, "insulin_units": bolus_value})
            continue

        if section_tag in {"basal", "temp_basal"}:
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                basal_rate = _to_float(attrs, ("rate", "value", "basal"))
                if timestamp is not None and basal_rate is not None:
                    insulin_rows.append({"timestamp": timestamp, "basal_rate": basal_rate})
            continue

        if section_tag in {"exercise", "work"}:
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                if timestamp is None:
                    continue
                intensity = _to_float(attrs, ("intensity", "value", "level"))
                duration = _to_float(attrs, ("duration", "minutes", "mins", "duration_minutes"))
                value = 0.5 if intensity is None else max(0.1, intensity / 10.0 if intensity > 1.0 else intensity)
                if duration and duration > 5:
                    end = timestamp + pd.Timedelta(minutes=float(duration))
                    _append_interval_samples(activity_rows, timestamp, end, "activity", float(value))
                else:
                    end = _to_end_timestamp(attrs)
                    if end is not None and end > timestamp:
                        _append_interval_samples(activity_rows, timestamp, end, "activity", float(value))
                    else:
                        activity_rows.append({"timestamp": timestamp, "activity": float(value)})
            continue

        if section_tag == "basis_steps":
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                steps = _to_float(attrs, ("value",))
                if timestamp is not None and steps is not None:
                    activity_rows.append({"timestamp": timestamp, "activity": min(float(steps) / 150.0, 1.0)})
            continue

        if section_tag == "basis_sleep":
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                start = _to_timestamp(attrs)
                end = _to_end_timestamp(attrs)
                if start is None:
                    continue
                if end is not None and end > start:
                    _append_interval_samples(activity_rows, start, end, "sleep_flag", 1.0)
                else:
                    activity_rows.append({"timestamp": start, "sleep_flag": 1.0})
            continue

        if section_tag == "stressors":
            for event in section.findall(".//event"):
                attrs = {_local_name(key): value for key, value in event.attrib.items()}
                timestamp = _to_timestamp(attrs)
                if timestamp is not None:
                    activity_rows.append({"timestamp": timestamp, "stress_score": 0.7})

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
