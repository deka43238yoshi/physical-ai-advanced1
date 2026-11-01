import csv
import json

import mujoco
import numpy as np
import pytest

from so101_dice_sim.cli import main
from so101_dice_sim.grasp import ROOT, run_grasp
from so101_dice_sim.observation import plan_observed_grasp, validate_position_offset
from so101_dice_sim.placement import placement_grasp_pose, placement_timing, validate_placement
from so101_dice_sim.position_batch import position_cases, run_position_batch
from so101_dice_sim.robot_check import JOINT_NAMES


@pytest.fixture(scope="module", params=[1, 2, 3])
def observed_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"observed-{request.param}")
    assert main([
        "--check-place", "--target-face", str(request.param), "--observe-position",
        "--initial-offset-mm", "5", "5", "--output-dir", str(root),
    ]) == 0
    directory, = root.iterdir()
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    with (directory / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    samples = [{
        key: (
            value if key == "phase" else value == "True" if value in ("True", "False")
            else int(value) if key == "top_face" and value else None if not value else float(value)
        )
        for key, value in row.items()
    } for row in rows]
    return directory, report, samples


def test_observed_run_uses_observation_and_actual_ik(observed_run):
    directory, report, samples = observed_run
    face = report["requested_face"]
    timing = placement_timing(face, observe_position=True)
    assert report["validation"]["passed"] and report["rotation_validation"]["passed"]
    assert report["grasp_validation"]["passed"]
    assert report["planning_mode"] == "observed_position"
    np.testing.assert_allclose(report["initial_qpos"][6:9], [.250, .020, .016])
    assert report["initial_qpos"][9:13] == [1, 0, 0, 0]
    assert report["equality_constraints"] == 0
    assert report["sample_count"] == len(samples) == round(timing.end_s / .002) + 1
    observation = report["observation_plan"]
    assert observation["observation_time_s"] == pytest.approx(2)
    assert observation["support_window_samples"] == 501
    np.testing.assert_allclose(observation["observed_dice_center_m"], [.25, .02, samples[1000]["dice_z_m"]])
    xy = placement_grasp_pose(face)[0] + [.005, .005]
    for phase in ("pregrasp", "lower", "lift"):
        np.testing.assert_allclose(report["planning"][phase]["target_position_m"][:2], xy)
    saved_plan = json.loads((directory / "observation_plan.json").read_text(encoding="utf-8"))
    assert saved_plan == observation
    assert saved_plan["robot_distance_query_cap_m"] == .05
    assert saved_plan["extra_hold_settle_s"] == 3
    assert report["grasp_validation"]["hold_start_s"] == 17
    assert report["grasp_validation"]["hold_samples"] == 1001
    assert samples[7000]["phase"] == "lift"
    assert samples[7001]["phase"] == "hold_settle"
    assert samples[8500]["phase"] == "hold"
    assert report["validation"]["final_window_start_s"] == timing.final_start_s
    assert report["validation"]["final_window_samples"] == 501
    assert report["validation"]["min_robot_clearance_m"] >= .02
    assert report["placement_plan"]["target_x_m"] == pytest.approx(.25 if face == 1 else .36)
    if face == 1:
        np.testing.assert_allclose(report["placement_plan"]["return_grasp_reference_xy_m"], xy)
    for i, sample in enumerate(samples):
        assert sample["time_s"] == pytest.approx(i * .002, abs=1e-9)
        assert sample["limits_ok"] and sample["unexpected_contacts"] == 0
        assert sample["min_arm_joint_margin_rad"] >= .1
        if 11 < sample["time_s"] <= timing.support_end_s + 1e-9:
            assert sample["gripper_target"] == 0
        if i >= round(timing.final_start_s / .002):
            assert sample["top_face"] == face
            assert sample["dice_table_contact"] and sample["dice_within_table"]
            assert sample["dice_robot_contact_count"] == 0
            assert sample["robot_clearance_m"] >= .02
            assert sample["dice_linear_speed_m_s"] < .001
            assert sample["dice_angular_speed_rad_s"] < .01
    result = validate_placement(samples, {}, target_face=face, observe_position=True)
    assert result["passed"]


def test_planner_reads_data_not_initialization_offset_and_preserves_state(observed_run):
    _, report, samples = observed_run
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    row = samples[1000]
    data.qpos[:6] = [row[f"{name}_qpos"] for name in JOINT_NAMES]
    data.qpos[6:9] = [.241, .018, row["dice_z_m"]]
    data.qpos[9:13] = [1, 0, 0, 0]
    data.time = 2
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    velocity = data.qvel.copy()
    xy, plan = plan_observed_grasp(model, data, samples[:1001], report["requested_face"])
    expected = placement_grasp_pose(report["requested_face"])[0] + [-.004, .003]
    np.testing.assert_allclose(xy, expected)
    np.testing.assert_allclose(plan["observed_dice_center_m"], data.xpos[model.body("dice").id])
    np.testing.assert_array_equal(before, data.qpos)
    np.testing.assert_array_equal(velocity, data.qvel)
    assert data.time == 2
    data.qpos[9:13] = [0, 1, 0, 0]
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="方位固定"):
        plan_observed_grasp(model, data, samples[:1001], report["requested_face"])
    data.qpos[9:13] = [1, 0, 0, 0]
    data.qpos[6] = .26
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="0.005"):
        plan_observed_grasp(model, data, samples[:1001], report["requested_face"])
    data.qpos[6] = .245
    mujoco.mj_forward(model, data)
    bad = list(samples[:1001])
    bad[750] = {**bad[750], "dice_table_force_n": 0}
    with pytest.raises(ValueError, match="支持"):
        plan_observed_grasp(model, data, bad, report["requested_face"])


@pytest.mark.parametrize("offset", [
    (.0051, 0), (0, -.0051), (float("nan"), 0), (0, float("inf")), (0,), (0, 0, 0),
])
def test_invalid_offset(offset, tmp_path):
    with pytest.raises(ValueError):
        validate_position_offset(offset)
    with pytest.raises(ValueError):
        run_grasp(tmp_path, place_face=2, observe_position=True, initial_offset_m=offset)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments", [
    ["--observe-position"], ["--check-grasp", "--observe-position"],
    ["--check-place", "--initial-offset-mm", "1", "2"],
])
def test_cli_modes(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_cli_rejects_out_of_range(tmp_path):
    assert main([
        "--check-place", "--observe-position", "--initial-offset-mm", "5.01", "0",
        "--output-dir", str(tmp_path),
    ]) == 2
    assert next(tmp_path.glob("*/error.json")).is_file()
    assert not list(tmp_path.glob("*/trajectory.csv"))


def test_observed_plan_failure_saves_partial_without_moving_arm(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("テスト用観測失敗")
    monkeypatch.setattr("so101_dice_sim.grasp.plan_observed_grasp", reject)
    assert main([
        "--check-place", "--observe-position", "--output-dir", str(tmp_path),
    ]) == 2
    abort = json.loads(next(tmp_path.glob("*/placement_abort.json")).read_text(encoding="utf-8"))
    assert abort["last_time_s"] == pytest.approx(2)
    assert abort["sample_count"] == 1001 and not abort["release_executed"]
    assert abort["final_state"]["gripper_target"] == .6
    assert not list(tmp_path.glob("*/result.json"))
    assert next(tmp_path.glob("*/aborted_trajectory.csv")).is_file()


def test_position_cases_reproducibility_and_exact_boundaries():
    cases = position_cases(42, 3)
    assert len(cases) == 12 and len(set(cases)) == 12
    assert set(cases[:9]) == {(x, y) for x in (-.005, 0, .005) for y in (-.005, 0, .005)}
    assert position_cases(42, 3) == cases
    assert position_cases(43, 3)[:9] == cases[:9]
    assert position_cases(43, 3)[9:] != cases[9:]
    assert all(abs(x) <= .005 and abs(y) <= .005 for x, y in cases)
    with pytest.raises(ValueError):
        position_cases(42, -1)


def test_batch_keeps_failures_and_runs_all_faces(tmp_path, monkeypatch):
    seen = []
    def fake_run(directory, **kwargs):
        seen.append(kwargs)
        face = kwargs["place_face"]
        if face == 1:
            raise ValueError("到達不能")
        return {
            "validation": {
                "passed": face == 3, "failure_reasons": [] if face == 3 else ["stationary"],
                "min_robot_clearance_m": .03,
            }, "final_state": {"top_face": face},
        }
    monkeypatch.setattr("so101_dice_sim.position_batch.run_grasp", fake_run)
    summary = run_position_batch(tmp_path, [(.003, -.002)], seed=42)
    assert summary["runs"] == 3 and summary["passed_count"] == 1 and not summary["passed"]
    assert [r["status"] for r in summary["results"]] == ["execution_error", "validation_failed", "passed"]
    assert all(k["observe_position"] and k["initial_offset_m"] == (.003, -.002) for k in seen)
    assert next(tmp_path.glob("*/error.json")).is_file()
    assert (tmp_path / "summary.csv").is_file() and (tmp_path / "summary.json").is_file()
