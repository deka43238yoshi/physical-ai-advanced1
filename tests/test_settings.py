from dataclasses import replace
import json
from pathlib import Path

import pytest

from so101_dice_sim.settings import load_settings


def test_default_settings(settings):
    assert settings.steps == 2500
    assert settings.window_steps == 500
    assert Path(settings.scene).is_file()


@pytest.mark.parametrize("changes", [
    {"duration_s": 0},
    {"duration_s": -1},
    {"duration_s": float("inf")},
    {"duration_s": float("nan")},
    {"duration_s": True},
    {"duration_s": "5"},
    {"timestep_s": 0},
    {"duration_s": 5.001},
    {"settle_window_s": 0.001},
    {"settle_window_s": 6},
    {"linear_speed_limit_m_s": -1},
    {"angular_speed_limit_rad_s": 0},
    {"height_tolerance_m": -1},
    {"face_tolerance_deg": 45},
    {"initial_position_m": [0, 0]},
    {"initial_position_m": [0, 0, float("nan")]},
    {"initial_position_m": [0, False, 1]},
    {"initial_quaternion_wxyz": [2, 0, 0, 0]},
    {"initial_quaternion_wxyz": [0, 0, 0, 0]},
    {"initial_quaternion_wxyz": None},
    {"scene": ""},
])
def test_invalid_settings(settings, changes):
    with pytest.raises(ValueError):
        replace(settings, **changes)


@pytest.mark.parametrize("payload", [
    [], {}, {"unknown": 1},
])
def test_invalid_config_schema(tmp_path, payload):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(config)
