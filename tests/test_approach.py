import csv
import json

import mujoco
import numpy as np
import pytest

from so101_dice_sim.approach import (
    ROOT, run_approach, solve_approach, tool_pose, validate_approach,
)
from so101_dice_sim.cli import main
from so101_dice_sim.robot_check import JOINT_NAMES


@pytest.fixture(scope="module")
def approach_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("approach")
    report = run_approach(directory)
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return directory, report, rows


def test_integrated_model_and_ik():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    assert model.neq == 0
    data = mujoco.MjData(model)
    original = data.qpos.copy()
    target = np.array([0.25, 0, 0.085])
    solution, planning = solve_approach(model, target)
    np.testing.assert_array_equal(data.qpos, original)
    assert planning["position_error_m"] < 0.0001
    assert planning["axis_error_deg"] < 0.1
    assert np.all(solution >= model.jnt_range[:6, 0])
    assert np.all(solution <= model.jnt_range[:6, 1])
    data.qpos[:6] = solution
    mujoco.mj_forward(model, data)
    position, axis = tool_pose(model, data)
    np.testing.assert_allclose(position, target, atol=0.0001)
    np.testing.assert_allclose(axis, [0, 0, -1], atol=0.002)


@pytest.mark.parametrize("target", [
    np.array([10.0, 0.0, 10.0]), np.array([np.nan, 0, 0]), np.zeros(2),
])
def test_unreachable_or_invalid_target_is_error(target):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    with pytest.raises(ValueError):
        solve_approach(model, target)


def test_physical_approach_and_dice_preservation(approach_run):
    _, report, rows = approach_run
    assert report["validation"]["passed"], report["validation"]
    assert report["sample_count"] == len(rows) == 4001
    assert report["evaluation_samples"] == 501
    assert report["duration_s"] == pytest.approx(8)
    assert report["table_top_m"] == 0
    assert report["dice_edge_m"] == pytest.approx(0.03)
    assert report["dice_mass_kg"] == pytest.approx(0.02)
    assert not report["warnings"]
    assert not report["unexpected_contact_pairs"]
    assert rows[0]["dice_table_contact"] == "False"
    assert rows[-1]["dice_table_contact"] == "True"
    assert float(rows[-1]["dice_z_m"]) < float(rows[0]["dice_z_m"])
    assert np.linalg.norm([
        float(rows[-1][f"tool_{a}_m"]) - float(rows[0][f"tool_{a}_m"])
        for a in "xyz"
    ]) > 0.1
    for index, row in enumerate(rows):
        assert float(row["time_s"]) == pytest.approx(index * 0.002, abs=1e-9)
        assert int(row["unexpected_contacts"]) == 0
        assert float(row["dice_xy_displacement_m"]) < 0.001
        assert float(row["gripper_qpos"]) == pytest.approx(0.6, abs=0.01)
        for joint in report["joints"]:
            q = float(row[f"{joint['name']}_qpos"])
            assert joint["range_rad"][0] - 1e-6 <= q <= joint["range_rad"][1] + 1e-6
            force = float(row[f"{joint['name']}_force"])
            assert abs(force) <= 2.940001
        if index >= 3500:
            assert float(row["position_error_m"]) < 0.002
            assert float(row["axis_error_deg"]) < 3
            assert float(row["max_joint_speed_rad_s"]) < 0.01
            assert float(row["dice_linear_speed_m_s"]) < 0.001
            assert float(row["dice_angular_speed_rad_s"]) < 0.01
            assert abs(float(row["dice_bottom_gap_m"])) <= 0.001
            assert row["dice_within_table"] == "True"
            assert row["top_face"] == "1"
    assert [float(rows[-1][f"{name}_target"]) for name in JOINT_NAMES] == (
        report["planning"]["target_joint_angles_rad"]
    )


def test_actual_pose_matches_recorded_joint_states(approach_run):
    _, _, rows = approach_run
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    for row in rows[::40]:
        data.qpos[:6] = [float(row[f"{name}_qpos"]) for name in JOINT_NAMES]
        data.qpos[6:9] = [float(row[f"dice_{axis}_m"]) for axis in "xyz"]
        data.qpos[9:13] = [float(row[f"dice_quat_{axis}"]) for axis in "wxyz"]
        mujoco.mj_forward(model, data)
        position, _ = tool_pose(model, data)
        np.testing.assert_allclose(
            position, [float(row[f"tool_{a}_m"]) for a in "xyz"], atol=1e-12
        )


@pytest.mark.parametrize("key,value,check", [
    ("unexpected_contacts", 1, "no_unexpected_contacts"),
    ("position_error_m", 0.002, "reached_position"),
    ("axis_error_deg", 3.0, "downward_tool_axis"),
    ("max_joint_speed_rad_s", 0.01, "arm_settled"),
    ("dice_table_contact", False, "dice_on_table"),
    ("dice_bottom_gap_m", 0.002, "dice_on_table"),
    ("dice_within_table", False, "dice_on_table"),
    ("dice_linear_speed_m_s", 0.001, "dice_settled"),
    ("dice_angular_speed_rad_s", 0.01, "dice_settled"),
    ("dice_xy_displacement_m", 0.002, "dice_not_displaced"),
    ("limits_ok", False, "joint_and_force_limits"),
    ("top_face", 2, "dice_top_face"),
    ("gripper_qpos", 0.5, "gripper_remains_open"),
    ("tool_x_m", float("nan"), "finite_values"),
])
def test_failed_window_sample_is_not_hidden(approach_run, key, value, check):
    _, report, _ = approach_run
    baseline = report["final_state"]
    samples = [
        {**baseline, "time_s": i * 0.002, "evaluation_window": i >= 3500}
        for i in range(4001)
    ]
    assert validate_approach(samples, {})["passed"]
    samples[3500][key] = value
    result = validate_approach(samples, {})
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_contact_during_movement_is_failure(approach_run):
    _, report, _ = approach_run
    baseline = report["final_state"]
    samples = [
        {**baseline, "time_s": i * 0.002, "evaluation_window": i >= 3500}
        for i in range(4001)
    ]
    samples[1700]["unexpected_contacts"] = 1
    assert not validate_approach(samples, {})["checks"]["no_unexpected_contacts"]
    assert not validate_approach(samples, {"mjWARN_BADQACC": 1})["passed"]
    assert not validate_approach(samples[:-1], {})["checks"]["sample_count"]
    samples[0]["time_s"] = 1
    assert not validate_approach(samples, {})["checks"]["simulation_times"]


def test_approach_report_and_no_overwrite(approach_run):
    directory, report, _ = approach_run
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == report
    before = (directory / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_approach(directory)
    assert (directory / "result.json").read_bytes() == before


def test_approach_cli(tmp_path):
    assert main(["--check-approach", "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["task"] == "so101_dice_pregrasp_approach"
    assert report["validation"]["passed"]


@pytest.mark.parametrize("extra", [
    ["--check-so101"], ["--random-position-runs", "2"],
])
def test_incompatible_modes(extra):
    with pytest.raises(SystemExit) as error:
        main(["--check-approach", *extra])
    assert error.value.code == 2
