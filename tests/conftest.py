from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_ohio_fixture(root: Path, patient_ids: tuple[str, ...] = ("540", "544"), rows: int = 240) -> Path:
    rng = np.random.default_rng(42)
    base_dir = root / "OhioT1DM"
    train_dir = base_dir / "OhioT1DM-training"
    test_dir = base_dir / "OhioT1DM-testing"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def build_xml(path: Path, patient_id: str, split: str) -> None:
        index = pd.date_range("2026-01-01", periods=rows, freq="5min")
        root_elem = ET.Element("patient", id=patient_id, split=split)
        offset = 0 if split == "train" else 8
        glucose = 118 + 12 * np.sin(np.linspace(0, 10, rows)) + rng.normal(0, 3, rows)
        glucose[120 + offset : 132 + offset] -= 42
        glucose[190 + offset : 205 + offset] -= 28
        glucose = np.clip(glucose, 42, 250)

        for ts, value in zip(index, glucose):
            ET.SubElement(root_elem, "glucose_level", ts=ts.isoformat(), value=f"{value:.1f}")

        for meal_idx in (36, 96, 156):
            meal_time = index[min(meal_idx + offset, rows - 1)]
            ET.SubElement(root_elem, "meal", ts=meal_time.isoformat(), carbs=f"{50 + (meal_idx % 20)}")
            ET.SubElement(root_elem, "bolus", ts=(meal_time - pd.Timedelta(minutes=10)).isoformat(), dose="4.5")

        ET.SubElement(root_elem, "basal", ts=index[0].isoformat(), rate="0.8")
        ET.SubElement(root_elem, "exercise", ts=index[min(100 + offset, rows - 1)].isoformat(), intensity="0.7", duration="40")
        ET.SubElement(
            root_elem,
            "sleep",
            ts=index[min(180 + offset, rows - 1)].isoformat(),
            end=index[min(220 + offset, rows - 1)].isoformat(),
        )
        ET.SubElement(root_elem, "stress", ts=index[min(140 + offset, rows - 1)].isoformat(), value="0.6")

        tree = ET.ElementTree(root_elem)
        tree.write(path, encoding="utf-8", xml_declaration=True)

    for patient_id in patient_ids:
        build_xml(train_dir / f"{patient_id}.xml", patient_id, "train")
        build_xml(test_dir / f"{patient_id}.xml", patient_id, "test")

    return base_dir
