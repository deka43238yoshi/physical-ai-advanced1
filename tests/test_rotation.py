import csv
import json

import numpy as np
import pytest

from so101_dice_sim.cli import main
from so101_dice_sim.grasp import run_grasp
from so101_dice_sim.rotation import validate_rotation


@pytest.fixture(scope="module")
def rotation_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rotation")
    report = run_grasp(directory, rotate_to_face=2)
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    samples = []
    for row in rows:
        sample = {}
        for key, value in row.items():
            if key == "phase":
                sample[key] = value
            elif value in ("True", "False"):
                sample[key] = value == "True"
            elif key == "top_face":
                sample[key] = int(value) if value else None
            else:
                sample[key] = float(value)
        samples.append(sample)
    return directory, report, samples


def test_physical_rotation_and_face(rotation_run):
    directory, report, samples = rotation_run
    assert report["task"] == "so101_fixed_grasp_rotate"
    assert report["requested_face"] == 2
    assert report["grasp_validation"]["passed"]
    assert report["rotation_validation"]["passed"], report["rotation_validation"]
    assert report["validation"]["passed"]
    assert report["equality_constraints"] == 0
    assert report["sample_count"] == len(samples) == 11001
    assert report["duration_s"] == pytest.approx(22)
    assert report["rotation_validation"]["hold_samples"] == 1001
    assert report["rotation_validation"]["rotation_samples"] == 3001
    assert report["rotation_validation"]["max_tilt_deg"] <= 5
    assert report["rotation_validation"]["max_relative_drift_m"] < 0.002
    assert samples[8000]["top_face"] == 1
    assert samples[-1]["top_face"] == 2
    assert validate_rotation(samples, {})["passed"]
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == report


def test_actual_rotation_path_samples(rotation_run):
    _, report, samples = rotation_run
    for i, sample in enumerate(samples):
        assert sample["time_s"] == pytest.approx(i * 0.002, abs=1e-9)
        assert sample["unexpected_contacts"] == 0
        assert sample["limits_ok"]
        for joint in report["joints"]:
            q = sample[f"{joint['name']}_qpos"]
            low, high = joint["range_rad"]
            assert low - 1e-6 <= q <= high + 1e-6
            low, high = joint["force_range_nm"]
            assert low - 1e-6 <= sample[f"{joint['name']}_force"] <= high + 1e-6
        if i >= 8000:
            assert sample["fixed_jaw_force_n"] >= 0.01
            assert sample["moving_jaw_force_n"] >= 0.01
            assert not sample["dice_table_contact"]
            assert sample["dice_bottom_gap_m"] >= 0.03
        if i >= 10000:
            assert sample["top_face"] == 2
            assert sample["tilt_deg"] <= 5
            assert sample["max_arm_speed_rad_s"] < 0.01
            assert sample["dice_linear_speed_m_s"] < 0.001
            assert sample["dice_angular_speed_rad_s"] < 0.01
    assert samples[-1]["wrist_flex_qpos"] - samples[8000]["wrist_flex_qpos"] == (
        pytest.approx(-np.pi / 2, abs=0.01)
    )


@pytest.mark.parametrize("index,key,value,check", [
    (9000, "fixed_jaw_force_n", 0, "both_jaws_through_rotation"),
    (9000, "moving_jaw_force_n", 0, "both_jaws_through_rotation"),
    (9000, "dice_table_contact", True, "off_table_through_rotation"),
    (9000, "dice_bottom_gap_m", 0.02, "off_table_through_rotation"),
    (9000, "dice_in_gripper_z_m", 1.0, "no_position_slip"),
    (9000, "unexpected_contacts", 1, "no_unexpected_contacts"),
    (10500, "top_face", 1, "requested_face_up"),
    (10500, "top_face", None, "requested_face_up"),
    (10500, "tilt_deg", 5.01, "requested_face_up"),
    (10500, "max_arm_speed_rad_s", 0.01, "hold_speeds"),
    (10500, "dice_linear_speed_m_s", 0.001, "hold_speeds"),
    (10500, "dice_angular_speed_rad_s", 0.01, "hold_speeds"),
    (10500, "wrist_flex_qpos", 1.0, "rotation_actually_executed"),
    (10500, "limits_ok", False, "joint_and_force_limits"),
    (10500, "min_arm_joint_margin_rad", 0.09, "joint_margin"),
    (9000, "dice_x_m", float("nan"), "finite_values"),
])
def test_rotation_single_bad_frame(rotation_run, index, key, value, check):
    samples = [dict(sample) for sample in rotation_run[2]]
    samples[index][key] = value
    result = validate_rotation(samples, {})
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_rotation_metadata_errors(rotation_run):
    samples = [dict(sample) for sample in rotation_run[2]]
    assert not validate_rotation(samples[:-1], {})["checks"]["sample_count"]
    assert not validate_rotation(samples[:-1], {})["checks"]["full_hold_window"]
    assert not validate_rotation(samples, {"warning": 1})["checks"]["no_mujoco_warnings"]
    samples[0]["time_s"] = 1
    assert not validate_rotation(samples, {})["checks"]["simulation_times"]


@pytest.mark.parametrize("face", [1, 3, 4, 5, 6, True])
def test_unsupported_faces_rejected(tmp_path, face):
    with pytest.raises(ValueError):
        run_grasp(tmp_path, rotate_to_face=face)
    assert not list(tmp_path.iterdir())


def test_rotation_cli(tmp_path):
    assert main(["--check-rotate", "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["validation"]["passed"]
    assert report["requested_face"] == 2


@pytest.mark.parametrize("extra", [
    ["--check-grasp"], ["--check-so101"], ["--check-approach"],
    ["--random-position-runs", "2"],
])
def test_rotation_mode_exclusive(extra):
    with pytest.raises(SystemExit) as error:
        main(["--check-rotate", *extra])
    assert error.value.code == 2
