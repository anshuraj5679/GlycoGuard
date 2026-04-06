from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


LIVE_WATCH_FILENAME = "live_watch_payload.json"


def live_watch_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / LIVE_WATCH_FILENAME


def write_live_watch_payload(base_dir: str | Path, payload: dict[str, object]) -> Path:
    path = live_watch_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_live_watch_payload(base_dir: str | Path, *, max_age_seconds: float = 300.0) -> dict[str, object] | None:
    path = live_watch_path(base_dir)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    updated_at = payload.get("updated_at")
    if not updated_at:
        return None

    try:
        age_seconds = (pd.Timestamp.now() - pd.Timestamp(updated_at)).total_seconds()
    except Exception:
        return None

    if age_seconds < 0 or age_seconds > max_age_seconds:
        return None

    return payload
