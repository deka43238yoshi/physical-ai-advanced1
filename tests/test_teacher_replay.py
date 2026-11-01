import json

import mujoco
import numpy as np
import pytest

from so101_dice_sim.grasp import capture_sample
from so101_dice_sim.robot_check import JOINT_NAMES, ROOT
from so101_dice_sim.teacher_replay import (
    action_table, gate_failures, held_action_index, interpolate_command, main, replay_episode,
    run_study,
)
from so101_dice_sim.visualization import Replay


@pytest.mark.parametrize("step,index", [(0, 0), (1, 0), (49, 0), (50, 1), (99, 1), (22499, 449)])
def test_zero_order_hold_uses_next_teacher_row_for_whole_period(step, index):
    assert held_action_index(step, 50) == index


@pytest.mark.parametrize("step,stride", [(-1, 50), (0, 0)])
def test_invalid_hold_schedule_rejected(step, stride):
    with pytest.raises(ValueError):
        held_action_index(step, stride)


def test_action_table_and_finite_validation():
    rows = [{f"{name}_target": str(i + j / 10) for j, name in enumerate(JOINT_NAMES)} for i in range(3)]
    replay = Replay({}, rows, np.arange(3) * .002, np.zeros((3, 13)), {})
    commands = action_table(replay)
    np.testing.assert_allclose(commands[1], np.arange(6) / 10 + 1)
    rows[0][f"{JOINT_NAMES[0]}_target"] = "nan"
    with pytest.raises(ValueError):
        action_table(replay)


def test_shared_recorder_does_not_change_physics_state():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[5] = .6
    data.qpos[6:9] = [.245, .015, .016]
    data.ctrl[:] = data.qpos[:6]
    mujoco.mj_forward(model, data)
    before = [value.copy() for value in (data.qpos, data.qvel, data.ctrl, data.qacc)]
    sample, unexpected = capture_sample(model, data, "initial", robot_geoms=[], distance_cap_m=.05)
    for actual, expected in zip((data.qpos, data.qvel, data.ctrl, data.qacc), before):
        np.testing.assert_array_equal(actual, expected)
    assert sample["time_s"] == 0
    assert sample["dice_z_m"] == .016
    assert sample["top_face"] == 1
    assert not unexpected
    data.qpos[0] = np.nan
    with pytest.raises(FloatingPointError):
        capture_sample(model, data, "initial")


def test_support_gate_refuses_missing_window_and_unknown_stage():
    assert gate_failures([], {}, "release") == ["supported_before_release"]
    with pytest.raises(ValueError):
        gate_failures([], {}, "unknown")


def test_existing_rotation_and_grasp_evaluators_are_used(monkeypatch):
    seen = []

    def grasp(samples, warnings, *, observe_position):
        seen.append(("grasp", len(samples), observe_position))
        return {"failure_reasons": ["held_without_slip"]}

    def rotation(samples, warnings, *, target_face, observe_position):
        seen.append(("rotation", len(samples), target_face, observe_position))
        return {"failure_reasons": ["no_position_slip", "hold_speeds"]}

    monkeypatch.setattr("so101_dice_sim.teacher_replay.validate_grasp", grasp)
    monkeypatch.setattr("so101_dice_sim.teacher_replay.validate_rotation", rotation)
    reasons = gate_failures([{}] * 12501, {}, "rotation_end")
    assert seen == [("grasp", 9501, True), ("rotation", 12501, 2, True)]
    assert reasons == ["grasp.held_without_slip", "rotation.no_position_slip", "rotation.hold_speeds"]


def test_invalid_replay_mode_rejected_before_physics(tmp_path):
    replay = Replay({}, [], np.empty(0), np.empty((0, 13)), {})
    with pytest.raises(ValueError, match="モード"):
        replay_episode(tmp_path, tmp_path, replay, np.zeros((0, 6)), "unrecognized")


def test_invalid_teacher_task_rejected_before_physics(tmp_path):
    replay = Replay({"task": "not_face2"}, [], np.empty(0), np.empty((0, 13)), {})
    with pytest.raises(ValueError, match="面2"):
        replay_episode(tmp_path, tmp_path, replay, np.zeros((0, 6)), "dataset_10hz")


def test_cli_persists_input_error(tmp_path):
    assert main(["--collection", str(tmp_path / "missing.json"), "--output-dir", str(tmp_path)]) == 2
    directory, = tmp_path.iterdir()
    report = json.loads((directory / "error.json").read_text(encoding="utf-8"))
    assert report["status"] == "error"


@pytest.mark.parametrize("smooth", [False, True])
def test_interpolation_is_continuous_bounded_and_reaches_endpoint(smooth):
    start = np.array([-.2, .4, 1., 0., -.5, .6])
    endpoint = np.array([.1, -.4, .5, .7, -.2, 0.])
    before = start.copy(), endpoint.copy()
    commands = np.array([
        interpolate_command(start, endpoint, step, 50, smooth=smooth) for step in range(1, 51)
    ])
    fraction = np.arange(1, 51) / 50
    expected_fraction = fraction**2 * (3 - 2 * fraction) if smooth else fraction
    np.testing.assert_allclose(commands, start + expected_fraction[:, None] * (endpoint - start))
    np.testing.assert_array_equal(commands[-1], endpoint)
    assert np.all(commands >= np.minimum(start, endpoint))
    assert np.all(commands <= np.maximum(start, endpoint))
    np.testing.assert_array_equal(start, before[0])
    np.testing.assert_array_equal(endpoint, before[1])
    next_endpoint = endpoint + .1
    np.testing.assert_allclose(
        interpolate_command(commands[-1], next_endpoint, 1, 50, smooth=smooth),
        endpoint + expected_fraction[0] * .1,
    )


def test_linear_interpolation_rate_after_bounded_endpoint():
    from so101_dice_sim.learning import bounded_action

    start = np.zeros(6)
    endpoint = bounded_action(np.full(6, .9), start, np.array([[-1., 1.]] * 6))
    commands = np.array([start, *[
        interpolate_command(start, endpoint, step, 50) for step in range(1, 51)
    ]])
    assert np.max(np.abs(np.diff(commands, axis=0))) <= .00100000001
    np.testing.assert_array_equal(commands[-1], np.full(6, .05))


@pytest.mark.parametrize("step,stride", [(0, 50), (51, 50), (1, 0)])
def test_invalid_interpolation_time_rejected(step, stride):
    with pytest.raises(ValueError):
        interpolate_command(np.zeros(6), np.ones(6), step, stride)


@pytest.mark.parametrize("endpoint", [np.ones(5), np.full(6, np.nan)])
def test_invalid_interpolation_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):
        interpolate_command(np.zeros(6), endpoint, 1, 50)


def test_study_rejects_empty_duplicate_and_invalid_modes_before_io(tmp_path):
    for modes in ((), ("endpoint_linear", "endpoint_linear"), ("invalid",)):
        with pytest.raises(ValueError):
            run_study(tmp_path, tmp_path / "missing.json", modes)


@pytest.mark.parametrize("requested,expected", [
    ([], ("source_500hz", "dataset_10hz", "dataset_10hz_bounded")),
    (["--modes", "endpoint_linear_bounded"], ("endpoint_linear_bounded",)),
])
def test_cli_preserves_default_and_forwards_selected_modes(tmp_path, monkeypatch, requested, expected):
    seen = []

    def study(directory, collection, modes):
        seen.append((directory.parent, collection, modes))
        return {"passed_count": 1, "runs": 1}

    monkeypatch.setattr("so101_dice_sim.teacher_replay.run_study", study)
    collection = tmp_path / "collection.json"
    assert main([
        "--collection", str(collection), "--output-dir", str(tmp_path), *requested,
    ]) == 0
    assert seen == [(tmp_path, collection, expected)]
