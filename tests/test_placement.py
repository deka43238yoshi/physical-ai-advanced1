import csv
import json

import mujoco
import numpy as np
import pytest

from so101_dice_sim.approach import solve_approach
from so101_dice_sim.cli import main
from so101_dice_sim.faces import identify_top_face
from so101_dice_sim.grasp import ROOT, run_grasp
from so101_dice_sim.placement import (
    PLACE_WRIST_ROLL_RAD, plan_placement, support_ready, validate_placement,
)


@pytest.fixture(scope="module")
def placement_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("placement")
    report = run_grasp(directory, place_face=2)
    return directory, report, read_samples(directory)


def read_samples(directory):
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
            elif value == "":
                sample[key] = None
            else:
                sample[key] = float(value)
        samples.append(sample)
    return samples


def test_fixed_end_to_end_placement(placement_run):
    _, report, samples = placement_run
    assert report["task"] == "so101_fixed_place_face_2"
    assert report["validation"]["passed"], report["validation"]
    assert report["grasp_validation"]["passed"]
    assert report["rotation_validation"]["passed"]
    assert report["placement_validation"]["passed"]
    assert report["requested_face"] == 2
    assert report["initial_qpos"][6:9] == [0.245, 0.015, 0.016]
    assert report["equality_constraints"] == 0
    assert report["duration_s"] == pytest.approx(42)
    assert report["sample_count"] == len(samples) == 21001
    assert report["runtime_overrides"]["gripper_force_range_nm"] == [-0.25, 0.25]
    assert report["runtime_overrides"]["upstream_gripper_force_range_nm"] == [-2.94, 2.94]
    assert report["placement_validation"]["final_window_samples"] == 501
    assert report["placement_validation"]["min_robot_clearance_m"] >= 0.02
    assert not report["warnings"]
    assert not report["unexpected_contact_pairs"]
    assert report["final_state"]["top_face"] == 2
    assert validate_placement(samples, {})["passed"]


def test_every_step_limits_and_support_before_open(placement_run):
    _, report, samples = placement_run
    assert support_ready(samples)
    assert samples[15500]["phase"] == "touch"
    assert samples[15501]["phase"] == "release"
    assert all(
        s["gripper_target"] == 0
        for s in samples if 30.5 <= s["time_s"] <= 31.0
    )
    for i, s in enumerate(samples):
        assert s["time_s"] == pytest.approx(i * 0.002, abs=1e-9)
        assert s["unexpected_contacts"] == 0
        assert s["limits_ok"]
        assert s["min_arm_joint_margin_rad"] >= 0.1
        for joint in report["joints"]:
            low, high = joint["range_rad"]
            assert low - 1e-6 <= s[f"{joint['name']}_qpos"] <= high + 1e-6
            low, high = joint["force_range_nm"]
            assert low - 1e-6 <= s[f"{joint['name']}_force"] <= high + 1e-6
        if i >= 20500:
            assert s["phase"] == "final_evaluation"
            assert s["top_face"] == 2
            assert s["tilt_deg"] <= 5
            assert s["dice_table_contact"]
            assert s["dice_table_force_n"] > 0.01
            assert s["dice_within_table"]
            assert abs(s["dice_bottom_gap_m"]) <= 0.001
            assert s["dice_robot_contact_count"] == 0
            assert s["robot_clearance_m"] >= 0.02
            assert abs(s["gripper_qpos"] - 0.6) < 0.01
            assert s["dice_linear_speed_m_s"] < 0.001
            assert s["dice_angular_speed_rad_s"] < 0.01


def test_placement_plan_does_not_change_physics_state(placement_run):
    _, report, samples = placement_run
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    row = samples[11000]
    data.qpos[:6] = [row[f"{j['name']}_qpos"] for j in report["joints"]]
    data.qpos[6:9] = [row[f"dice_{axis}_m"] for axis in "xyz"]
    data.qpos[9:13] = [row[f"dice_quat_{axis}"] for axis in "wxyz"]
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    before_velocity = data.qvel.copy()
    phases, plan = plan_placement(model, data)
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(data.qvel, before_velocity)
    assert [name for name, _, _ in phases] == [
        "place_above", "touch", "release", "retreat", "final_settle"
    ]
    assert plan["ik"]["touch"]["residual_norm"] < 1e-6
    for row in samples[20500::100]:
        data.qpos[:6] = [row[f"{j['name']}_qpos"] for j in report["joints"]]
        data.qpos[6:9] = [row[f"dice_{axis}_m"] for axis in "xyz"]
        data.qpos[9:13] = [row[f"dice_quat_{axis}"] for axis in "wxyz"]
        mujoco.mj_forward(model, data)
        dice = model.body("dice").id
        face, tilt = identify_top_face(data.xmat[dice].reshape(3, 3), 5)
        assert face == 2
        assert tilt == pytest.approx(row["tilt_deg"], abs=1e-6)
        extent = np.abs(data.xmat[dice].reshape(3, 3)) @ np.full(3, 0.015)
        assert data.xpos[dice, 2] - extent[2] == pytest.approx(
            row["dice_bottom_gap_m"], abs=1e-12
        )


@pytest.mark.parametrize("key,value,check", [
    ("top_face", 1, "requested_face_up"),
    ("top_face", None, "requested_face_up"),
    ("tilt_deg", 5.01, "requested_face_up"),
    ("dice_table_contact", False, "supported_on_table"),
    ("dice_table_force_n", 0, "supported_on_table"),
    ("dice_within_table", False, "supported_on_table"),
    ("dice_bottom_gap_m", 0.002, "supported_on_table"),
    ("dice_robot_contact_count", 1, "no_robot_contact"),
    ("robot_clearance_m", 0.019, "robot_retracted"),
    ("robot_clearance_m", None, "robot_retracted"),
    ("gripper_qpos", 0.5, "gripper_open"),
    ("dice_linear_speed_m_s", 0.001, "stationary"),
    ("dice_angular_speed_rad_s", 0.01, "stationary"),
    ("unexpected_contacts", 1, "no_unexpected_contacts"),
    ("limits_ok", False, "joint_and_force_limits"),
    ("min_arm_joint_margin_rad", 0.09, "joint_margin"),
    ("dice_x_m", float("nan"), "finite_values"),
])
def test_single_final_frame_failure(placement_run, key, value, check):
    samples = list(placement_run[2])
    samples[20750] = {**samples[20750], key: value}
    result = validate_placement(samples, {})
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_support_requires_full_window_and_force(placement_run):
    samples = list(placement_run[2])
    assert support_ready(samples)
    samples[15300] = {**samples[15300], "dice_table_force_n": 0}
    assert not support_ready(samples)
    assert not validate_placement(samples, {})["checks"]["supported_before_release"]
    assert not support_ready(samples[:15300])


def test_window_times_warnings(placement_run):
    samples = list(placement_run[2])
    assert not validate_placement(samples[:-1], {})["checks"]["sample_count"]
    assert not validate_placement(samples[:-1], {})["checks"]["full_final_window"]
    assert not validate_placement(samples, {"warning": 1})["checks"]["no_mujoco_warnings"]
    samples[0] = {**samples[0], "time_s": 1}
    assert not validate_placement(samples, {})["checks"]["simulation_times"]


def test_release_is_aborted_without_verified_support(tmp_path, monkeypatch):
    monkeypatch.setattr("so101_dice_sim.grasp.support_ready", lambda samples, **kwargs: False)
    assert main(["--check-place", "--output-dir", str(tmp_path)]) == 2
    abort = json.loads(next(tmp_path.glob("*/placement_abort.json")).read_text(encoding="utf-8"))
    assert not abort["release_executed"]
    assert abort["last_time_s"] == pytest.approx(31)
    assert abort["sample_count"] == 15501
    assert abort["final_state"]["gripper_target"] == 0
    assert next(tmp_path.glob("*/aborted_trajectory.csv")).is_file()
    assert not list(tmp_path.glob("*/result.json"))
    assert next(tmp_path.glob("*/error.json")).is_file()


def test_placement_report_and_no_overwrite(placement_run):
    directory, report, _ = placement_run
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == report
    original = (directory / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_grasp(directory, place_face=2)
    assert (directory / "result.json").read_bytes() == original


@pytest.mark.parametrize("face", [0, 7, 4, 5, 6, True, 1.0, "1"])
def test_unsupported_placement_face(tmp_path, face):
    with pytest.raises(ValueError):
        run_grasp(tmp_path, place_face=face)
    assert not list(tmp_path.iterdir())


def test_cross_grip_ik_and_invalid_roll():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    q, result = solve_approach(
        model, np.array([0.25, 0, 0.02]), wrist_roll=PLACE_WRIST_ROLL_RAD
    )
    assert q[4] == pytest.approx(-1.52, abs=0.001)
    assert result["position_error_m"] < 0.0001
    for invalid in [float("nan"), 10.0]:
        with pytest.raises(ValueError):
            solve_approach(model, np.array([0.25, 0, 0.02]), wrist_roll=invalid)


def test_placement_cli(tmp_path):
    assert main(["--check-place", "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["validation"]["passed"]
    assert report["task"] == "so101_fixed_place_face_2"


@pytest.mark.parametrize("extra", [
    ["--check-grasp"], ["--check-rotate"], ["--check-so101"], ["--check-approach"],
    ["--random-position-runs", "2"],
])
def test_placement_modes_exclusive(extra):
    with pytest.raises(SystemExit) as error:
        main(["--check-place", *extra])
    assert error.value.code == 2


@pytest.fixture(scope="module")
def face_one_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("placement-face-one")
    assert main(["--check-place", "--target-face", "1", "--output-dir", str(root)]) == 0
    directory, = root.iterdir()
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return directory, report, read_samples(directory)


def test_face_one_real_placement_and_unchanged_initial_condition(face_one_run, placement_run):
    _, report, samples = face_one_run
    assert report["task"] == "so101_fixed_place_face_1"
    assert report["requested_face"] == 1
    assert report["initial_qpos"] == placement_run[1]["initial_qpos"]
    assert report["validation"]["passed"]
    assert report["grasp_validation"]["passed"]
    assert report["rotation_validation"]["passed"]
    assert report["equality_constraints"] == 0
    assert report["rotation_plan"]["delta_rad"] == 0
    assert report["rotation_validation"]["required_wrist_flex_delta_rad"] == 0
    assert report["sample_count"] == 21001
    assert report["placement_plan"]["target_x_m"] == .245
    assert report["placement_validation"]["final_window_samples"] == 501
    assert report["placement_validation"]["min_robot_clearance_m"] >= .02
    assert all(
        s["gripper_target"] == 0
        for s in samples if 11 < s["time_s"] <= 31 + 1e-9
    )
    assert samples[15500]["phase"] == "touch"
    assert samples[15501]["phase"] == "release"
    assert all(s["top_face"] == 1 for s in samples[20500:])
    assert support_ready(samples, target_face=1)
    assert not support_ready(samples, target_face=2)
    assert validate_placement(samples, {}, target_face=1)["passed"]
    assert not validate_placement(samples, {}, target_face=2)["passed"]


@pytest.mark.parametrize("index,check", [
    (15300, "supported_before_release"), (20750, "requested_face_up"),
])
def test_face_one_rejects_wrong_face_in_single_frame(face_one_run, index, check):
    samples = list(face_one_run[2])
    samples[index] = {**samples[index], "top_face": 2}
    result = validate_placement(samples, {}, target_face=1)
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_face_one_hold_is_not_a_rotation(face_one_run):
    from so101_dice_sim.rotation import validate_rotation

    samples = face_one_run[2][:11001]
    assert validate_rotation(samples, {}, target_face=1)["passed"]
    assert not validate_rotation(samples, {}, target_face=2)["passed"]
    altered = list(samples)
    altered[10500] = {**samples[10500], "wrist_flex_qpos": samples[10500]["wrist_flex_qpos"] + .02}
    assert not validate_rotation(altered, {}, target_face=1)["checks"]["rotation_actually_executed"]


def test_face_one_plan_does_not_change_physics(face_one_run):
    _, _, samples = face_one_run
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    from so101_dice_sim.robot_check import JOINT_NAMES

    row = samples[11000]
    data.qpos[:6] = [row[f"{name}_qpos"] for name in JOINT_NAMES]
    data.qpos[6:9] = [row[f"dice_{axis}_m"] for axis in "xyz"]
    data.qpos[9:13] = [row[f"dice_quat_{axis}"] for axis in "wxyz"]
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    phases, plan = plan_placement(model, data, target_face=1)
    np.testing.assert_array_equal(data.qpos, before)
    assert data.time == 0
    assert plan["requested_face"] == 1
    assert plan["ik"]["touch"]["position_error_m"] < .0001
    assert phases[2][1][5] == .6
    for row in samples[20500::100]:
        data.qpos[9:13] = [row[f"dice_quat_{axis}"] for axis in "wxyz"]
        mujoco.mj_forward(model, data)
        assert identify_top_face(data.xmat[model.body("dice").id].reshape(3, 3), 5)[0] == 1


@pytest.mark.parametrize("face", [4, 5, 6])
def test_unimplemented_target_cli_is_explicit_error(tmp_path, face, capsys):
    assert main(["--check-place", "--target-face", str(face), "--output-dir", str(tmp_path)]) == 2
    assert "調査段階" in capsys.readouterr().err
    assert not list(tmp_path.glob("*/result.json"))
    assert not list(tmp_path.glob("*/trajectory.csv"))
    error = json.loads(next(tmp_path.glob("*/error.json")).read_text(encoding="utf-8"))
    assert error["status"] == "error"


@pytest.mark.parametrize("arguments", [
    ["--target-face", "1"], ["--check-grasp", "--target-face", "1"],
    ["--check-place", "--target-face", "0"], ["--check-place", "--target-face", "7"],
])
def test_target_face_cli_requirements(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_survey_six_faces_without_physics(placement_run, tmp_path, monkeypatch):
    from so101_dice_sim.feasibility import survey_faces

    source = placement_run[0]
    originals = {name: (source / name).read_bytes() for name in ("trajectory.csv", "result.json")}

    def forbid_step(*args, **kwargs):
        pytest.fail("幾何調査中に物理ステップを実行しました")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    report = survey_faces(source, tmp_path)
    assert [f["target_face"] for f in report["faces"]] == list(range(1, 7))
    assert report["faces"][0]["kinematic_clear_candidates"] > 0
    assert report["faces"][1]["kinematic_clear_candidates"] > 0
    for face in report["faces"]:
        assert face["physical_placement_verified"] == (face["target_face"] == 2)
        for candidate in face["candidates"]:
            assert candidate["hold_geometry"]["top_face"] == face["target_face"]
            if not candidate["blockers"]:
                assert candidate["touch_geometry"]["top_face"] == face["target_face"]
                assert candidate["rotation_path_min_bottom_gap_m"] >= .03
                assert not candidate["lower_path_unexpected_contacts"]
                assert not candidate["touch_geometry"]["unexpected_contact_pairs"]
    assert json.loads((tmp_path / "feasibility.json").read_text(encoding="utf-8")) == report
    assert len(list(csv.reader((tmp_path / "summary.csv").open(encoding="utf-8-sig")))) == 7
    for name, content in originals.items():
        assert (source / name).read_bytes() == content


@pytest.mark.parametrize("key,value", [
    ("task", "so101_fixed_place_face_1"), ("mujoco_version", "0"),
    ("source_commit", "changed"), ("validation", {"passed": False}),
])
def test_survey_rejects_wrong_source(tmp_path, key, value):
    from so101_dice_sim.feasibility import survey_faces

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    report = {
        "task": "so101_fixed_place_face_2", "source_commit": "f3b6013629c7b94f8ee93fd977b161fa8350f9bc",
        "mujoco_version": mujoco.__version__, "validation": {"passed": True},
    }
    report[key] = value
    (source / "result.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        survey_faces(source, output)
    assert not list(output.iterdir())


@pytest.fixture(scope="module")
def face_three_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("placement-face-three")
    assert main(["--check-place", "--target-face", "3", "--output-dir", str(root)]) == 0
    directory, = root.iterdir()
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return directory, report, read_samples(directory)


def test_face_three_full_pipeline(face_three_run, placement_run):
    _, report, samples = face_three_run
    assert report["task"] == "so101_fixed_place_face_3"
    assert report["requested_face"] == 3
    assert report["initial_qpos"] == placement_run[1]["initial_qpos"]
    assert report["duration_s"] == pytest.approx(46)
    assert report["sample_count"] == len(samples) == 23001
    for key in ("validation", "grasp_validation", "rotation_validation", "placement_validation"):
        assert report[key]["passed"], report[key]
    assert report["equality_constraints"] == 0
    assert not report["warnings"] and not report["unexpected_contact_pairs"]
    assert report["planning"]["lower"]["target_position_m"] == [.26, .02, .02]
    assert abs(report["planning"]["lower"]["target_joint_angles_rad"][4]) < .001
    assert report["runtime_overrides"]["gripper_force_range_nm"] == [-.25, .25]
    assert report["rotation_validation"]["rotation_samples"] == 5001
    assert report["rotation_validation"]["hold_samples"] == 1001
    assert report["placement_validation"]["final_window_samples"] == 501
    assert report["placement_plan"]["target_x_m"] == .36
    assert report["placement_plan"]["timing_s"]["end_s"] == 46


def test_face_three_sequence_and_every_step(face_three_run):
    _, report, samples = face_three_run
    assert [s["phase"] for s in report["phases"]] == [
        "settle", "pregrasp", "lower", "close", "lift", "hold", "wrist_twist",
        "rotate", "rotation_hold", "place_above", "touch", "release", "retreat", "final_settle",
    ]
    assert samples[10000]["phase"] == "wrist_twist"
    assert samples[10001]["phase"] == "rotate"
    assert samples[12000]["phase"] == "rotation_hold"
    assert samples[13000]["phase"] == "rotation_hold"
    assert samples[17500]["phase"] == "touch"
    assert samples[17501]["phase"] == "release"
    for i, s in enumerate(samples):
        assert s["time_s"] == pytest.approx(i * .002, abs=1e-9)
        assert s["unexpected_contacts"] == 0
        assert s["limits_ok"] and s["min_arm_joint_margin_rad"] >= .1
        for joint in report["joints"]:
            name = joint["name"]
            low, high = joint["range_rad"]
            assert low - 1e-6 <= s[f"{name}_qpos"] <= high + 1e-6
            low, high = joint["force_range_nm"]
            assert low - 1e-6 <= s[f"{name}_force"] <= high + 1e-6
        if 11 < s["time_s"] <= 35 + 1e-9:
            assert s["gripper_target"] == 0
        if i >= 22500:
            assert s["phase"] == "final_evaluation"
            assert s["top_face"] == 3 and s["tilt_deg"] <= 5
            assert s["dice_table_contact"] and s["dice_table_force_n"] > .01
            assert s["dice_within_table"] and abs(s["dice_bottom_gap_m"]) <= .001
            assert s["dice_robot_contact_count"] == 0
            assert s["robot_clearance_m"] >= .02
            assert abs(s["gripper_qpos"] - .6) < .01
            assert s["dice_linear_speed_m_s"] < .001
            assert s["dice_angular_speed_rad_s"] < .01
    assert support_ready(samples, target_face=3)
    assert not support_ready(samples, target_face=2)
    assert not validate_placement(samples, {}, target_face=2)["passed"]


@pytest.mark.parametrize("joint", ["wrist_roll", "wrist_flex"])
def test_face_three_both_rotations_required(face_three_run, joint):
    from so101_dice_sim.rotation import validate_rotation

    samples = list(face_three_run[2][:13001])
    assert validate_rotation(samples, {}, target_face=3)["passed"]
    row = samples[12500]
    samples[12500] = {**row, f"{joint}_qpos": samples[8000][f"{joint}_qpos"]}
    result = validate_rotation(samples, {}, target_face=3)
    assert not result["passed"]
    assert "rotation_actually_executed" in result["failure_reasons"]


@pytest.mark.parametrize("index,key,value,check", [
    (17300, "top_face", 2, "supported_before_release"),
    (17300, "dice_table_force_n", 0, "supported_before_release"),
    (17300, "dice_angular_speed_rad_s", .01, "supported_before_release"),
    (22750, "top_face", 2, "requested_face_up"),
    (22750, "tilt_deg", 5.01, "requested_face_up"),
    (22750, "robot_clearance_m", .019, "robot_retracted"),
    (22750, "dice_robot_contact_count", 1, "no_robot_contact"),
    (22750, "dice_table_force_n", 0, "supported_on_table"),
    (22750, "dice_linear_speed_m_s", .001, "stationary"),
    (22750, "gripper_qpos", .5, "gripper_open"),
])
def test_face_three_single_failure(face_three_run, index, key, value, check):
    samples = list(face_three_run[2])
    samples[index] = {**samples[index], key: value}
    result = validate_placement(samples, {}, target_face=3)
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_face_three_plan_and_face_geometry_without_state_mutation(face_three_run):
    from so101_dice_sim.robot_check import JOINT_NAMES

    _, _, samples = face_three_run
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)

    def restore(row):
        data.qpos[:6] = [row[f"{name}_qpos"] for name in JOINT_NAMES]
        data.qpos[6:9] = [row[f"dice_{axis}_m"] for axis in "xyz"]
        data.qpos[9:13] = [row[f"dice_quat_{axis}"] for axis in "wxyz"]
        mujoco.mj_forward(model, data)

    restore(samples[13000])
    before = data.qpos.copy()
    before_velocity = data.qvel.copy()
    phases, plan = plan_placement(model, data, target_face=3)
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(data.qvel, before_velocity)
    assert data.time == 0
    assert plan["requested_face"] == 3
    assert plan["ik"]["touch"]["residual_norm"] < 1e-6
    assert phases[2][1][5] == .6
    for row in samples[22500::100]:
        restore(row)
        dice = model.body("dice").id
        r = data.xmat[dice].reshape(3, 3)
        face, tilt = identify_top_face(r, 5)
        assert face == 3 and tilt == pytest.approx(row["tilt_deg"], abs=1e-6)
        extent = np.abs(r) @ np.full(3, .015)
        assert data.xpos[dice, 2] - extent[2] == pytest.approx(row["dice_bottom_gap_m"], abs=1e-12)


def test_face_three_aborts_before_release_at_new_support_time(tmp_path, monkeypatch):
    monkeypatch.setattr("so101_dice_sim.grasp.support_ready", lambda samples, **kwargs: False)
    assert main([
        "--check-place", "--target-face", "3", "--output-dir", str(tmp_path),
    ]) == 2
    abort = json.loads(next(tmp_path.glob("*/placement_abort.json")).read_text(encoding="utf-8"))
    assert not abort["release_executed"]
    assert abort["last_time_s"] == pytest.approx(35)
    assert abort["sample_count"] == 17501
    assert abort["final_state"]["gripper_target"] == 0
    assert next(tmp_path.glob("*/error.json")).is_file()
    assert not list(tmp_path.glob("*/result.json"))
