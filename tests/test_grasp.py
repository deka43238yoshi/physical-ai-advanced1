import csv
import json

import mujoco
import numpy as np
import pytest

from so101_dice_sim.cli import main
from so101_dice_sim.grasp import (
    ROOT, contact_state, run_grasp, validate_grasp,
)


@pytest.fixture(scope="module")
def grasp_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("grasp")
    report = run_grasp(directory)
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return directory, report, rows


def typed_rows(rows):
    result = []
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
        result.append(sample)
    return result


def test_actual_grasp_lift_hold(grasp_run):
    _, report, rows = grasp_run
    assert report["validation"]["passed"], report["validation"]
    assert all(report["validation"]["checks"].values())
    assert report["sample_count"] == len(rows) == 8001
    assert report["duration_s"] == pytest.approx(16)
    assert report["equality_constraints"] == 0
    assert not report["warnings"]
    assert not report["unexpected_contact_pairs"]
    assert report["initial_qpos"][6:9] == [0.235, 0, 0.016]
    assert report["runtime_overrides"]["gripper_force_range_nm"] == [-0.2, 0.2]
    assert report["runtime_overrides"]["upstream_gripper_force_range_nm"] == [-2.94, 2.94]
    assert report["validation"]["min_arm_joint_margin_rad"] >= 0.1
    assert report["validation"]["hold_samples"] == 1001
    assert report["validation"]["min_clearance_m"] >= 0.03
    assert report["validation"]["max_relative_drift_m"] < 0.002
    assert report["validation"]["min_fixed_jaw_force_n"] >= 0.01
    assert report["validation"]["min_moving_jaw_force_n"] >= 0.01
    assert report["final_state"]["gripper_qpos"] > 0.05
    assert report["final_state"]["gripper_target"] == 0
    assert report["final_state"]["dice_z_m"] - report["initial_qpos"][8] >= 0.035
    assert validate_grasp(typed_rows(rows), {})["passed"]


def test_every_step_joint_limits_and_torque(grasp_run):
    _, report, rows = grasp_run
    for index, row in enumerate(rows):
        assert float(row["time_s"]) == pytest.approx(index * 0.002, abs=1e-9)
        assert row["limits_ok"] == "True"
        assert int(row["unexpected_contacts"]) == 0
        for joint in report["joints"]:
            q = float(row[f"{joint['name']}_qpos"])
            low, high = joint["range_rad"]
            assert low - 1e-6 <= q <= high + 1e-6
            torque = float(row[f"{joint['name']}_force"])
            low, high = joint["force_range_nm"]
            assert low - 1e-6 <= torque <= high + 1e-6


def test_dice_height_and_gripper_frame_from_actual_state(grasp_run):
    _, report, rows = grasp_run
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    assert model.neq == 0
    data = mujoco.MjData(model)
    for row in rows[::50]:
        data.qpos[:6] = [float(row[f"{j['name']}_qpos"]) for j in report["joints"]]
        data.qpos[6:9] = [float(row[f"dice_{axis}_m"]) for axis in "xyz"]
        data.qpos[9:13] = [float(row[f"dice_quat_{axis}"]) for axis in "wxyz"]
        mujoco.mj_forward(model, data)
        dice = model.body("dice").id
        gripper = model.body("gripper").id
        extent = np.abs(data.xmat[dice].reshape(3, 3)) @ np.full(3, 0.015)
        assert float(row["dice_bottom_gap_m"]) == pytest.approx(
            data.xpos[dice, 2] - extent[2], abs=1e-12
        )
        relative = data.xmat[gripper].reshape(3, 3).T @ (
            data.xpos[dice] - data.xpos[gripper]
        )
        np.testing.assert_allclose(
            relative, [float(row[f"dice_in_gripper_{a}_m"]) for a in "xyz"], atol=1e-12
        )


@pytest.mark.parametrize("key,value,check", [
    ("fixed_jaw_force_n", 0.0, "both_jaws_during_hold"),
    ("moving_jaw_force_n", 0.0, "both_jaws_during_hold"),
    ("dice_table_contact", True, "off_table_during_hold"),
    ("dice_bottom_gap_m", 0.029, "off_table_during_hold"),
    ("dice_z_m", 0.04, "lifted_from_initial_height"),
    ("unexpected_contacts", 1, "no_unexpected_contacts"),
    ("min_arm_joint_margin_rad", 0.05, "joint_margin"),
    ("limits_ok", False, "joint_and_force_limits"),
    ("dice_in_gripper_z_m", 0.0, "held_without_slip"),
    ("dice_linear_speed_m_s", 0.001, "hold_speeds"),
    ("dice_angular_speed_rad_s", 0.01, "hold_speeds"),
    ("max_arm_speed_rad_s", 0.01, "hold_speeds"),
    ("dice_x_m", float("nan"), "finite_values"),
])
def test_single_hold_frame_failure(grasp_run, key, value, check):
    samples = typed_rows(grasp_run[2])
    samples[7500][key] = value
    result = validate_grasp(samples, {})
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_initial_and_closing_checks(grasp_run):
    samples = typed_rows(grasp_run[2])
    samples[600]["dice_table_contact"] = False
    assert not validate_grasp(samples, {})["checks"]["initial_dice_settled"]
    samples[5300]["fixed_jaw_force_n"] = 0
    assert not validate_grasp(samples, {})["checks"]["closed_on_both_jaws"]
    assert not validate_grasp(samples, {"warning": 1})["checks"]["no_mujoco_warnings"]
    assert not validate_grasp(samples[:-1], {})["checks"]["full_hold_window"]
    samples[0]["time_s"] = 1
    assert not validate_grasp(samples, {})["checks"]["simulation_times"]


def test_contact_classifier_rejects_arm_table():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:6] = [0, 0.5, 0.5, 0.5, 0, 0.6]
    mujoco.mj_forward(model, data)
    state, pairs = contact_state(model, data)
    assert pairs
    assert state["unexpected_contacts"] > 0


def test_report_and_no_overwrite(grasp_run):
    directory, report, _ = grasp_run
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == report
    before = (directory / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_grasp(directory)
    assert (directory / "result.json").read_bytes() == before


def test_grasp_cli(tmp_path):
    assert main(["--check-grasp", "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["task"] == "so101_fixed_grasp_lift"
    assert report["validation"]["passed"]


@pytest.mark.parametrize("extra", [
    ["--check-so101"], ["--check-approach"], ["--random-position-runs", "2"],
])
def test_grasp_cli_rejects_other_modes(extra):
    with pytest.raises(SystemExit) as error:
        main(["--check-grasp", *extra])
    assert error.value.code == 2
