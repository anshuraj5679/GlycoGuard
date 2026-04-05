from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_pickle(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    target = Path(path)
    with target.open("rb") as handle:
        return pickle.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)

