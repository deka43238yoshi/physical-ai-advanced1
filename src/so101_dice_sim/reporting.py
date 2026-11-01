import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from .faces import FACE_AXES
from .settings import Settings
from .simulation import Sample, SimulationResult


def create_run_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_row(sample: Sample) -> dict:
    row = {
        "time_s": sample.time_s,
        "table_contact": sample.table_contact,
        "bottom_gap_m": sample.bottom_gap_m,
        "within_table": sample.within_table,
        "top_face": sample.top_face,
        "tilt_deg": sample.tilt_deg,
    }
    for prefix, labels, values in (
        ("position", "xyz", sample.position_m),
        ("quaternion", "wxyz", sample.quaternion_wxyz),
        ("linear_velocity", "xyz", sample.linear_velocity_m_s),
        ("angular_velocity", "xyz", sample.angular_velocity_rad_s),
    ):
        row.update({f"{prefix}_{label}": v for label, v in zip(labels, values)})
    return row


def save_result(
    directory: Path, settings: Settings, result: SimulationResult, validation: dict
) -> None:
    with (directory / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        rows = (csv_row(sample) for sample in result.samples)
        first = next(rows)
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
    write_json(directory / "result.json", {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "mujoco": version("mujoco"),
            "numpy": version("numpy"),
            "so101_dice_sim": version("so101-dice-sim"),
        },
        "units": {
            "length": "m", "time": "s", "mass": "kg",
            "linear_velocity": "m/s", "angular_velocity": "rad/s",
            "tilt": "deg", "quaternion": "wxyz",
            "vectors_frame": "world",
        },
        "settings": asdict(settings),
        "model": result.model,
        "face_axes_local": FACE_AXES,
        "steps": len(result.samples) - 1,
        "sample_count": len(result.samples),
        "warnings": result.warnings,
        "validation": validation,
    })
