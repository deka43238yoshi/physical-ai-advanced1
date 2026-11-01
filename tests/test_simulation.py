from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from so101_dice_sim.simulation import run_simulation, validate_result


def test_physical_settling(settings, settled):
    validation = validate_result(settled, settings)
    assert validation["passed"], validation
    assert all(validation["checks"].values())
    assert len(settled.samples) == 2501
    assert validation["window_samples"] == 501
    assert validation["max_linear_speed_m_s"] < 0.001
    assert validation["max_angular_speed_rad_s"] < 0.01
    assert validation["max_abs_bottom_gap_m"] <= 0.001
    assert settled.model["dice_edge_m"] == pytest.approx(0.03)
    assert settled.model["dice_mass_kg"] == pytest.approx(0.02)
    assert settled.model["table_top_m"] == pytest.approx(0.4)
    assert settled.samples[0].top_face is None
    assert not settled.samples[0].table_contact
    assert settled.samples[-1].top_face in range(1, 7)
    assert settled.samples[-1].time_s == pytest.approx(5)
    assert settled.samples[-1].position_m[2] < settings.initial_position_m[2]
    np.testing.assert_allclose(
        settled.samples[0].quaternion_wxyz, settings.initial_quaternion_wxyz
    )
    assert not np.allclose(
        settled.samples[0].quaternion_wxyz, settled.samples[-1].quaternion_wxyz
    )


def test_every_sample_geometry_and_timing(settings, settled):
    half_size = np.full(3, 0.015)
    for step, sample in enumerate(settled.samples):
        assert sample.time_s == pytest.approx(step * settings.timestep_s, abs=1e-9)
        assert math.hypot(*sample.quaternion_wxyz) == pytest.approx(1)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, np.array(sample.quaternion_wxyz))
        extent = np.abs(matrix.reshape(3, 3)) @ half_size
        assert sample.bottom_gap_m == pytest.approx(
            sample.position_m[2] - extent[2] - 0.4
        )


def test_world_velocity_consistent_with_motion(settings, settled):
    before, sample, after = settled.samples[9:12]
    derivative = (
        np.asarray(after.position_m) - np.asarray(before.position_m)
    ) / (2 * settings.timestep_s)
    np.testing.assert_allclose(
        sample.linear_velocity_m_s, derivative, atol=0.011
    )
    assert sample.linear_velocity_m_s[2] < -0.1


@pytest.mark.parametrize("change,failed_check", [
    ({"linear_velocity_m_s": (0.001, 0, 0)}, "linear_speed"),
    ({"angular_velocity_rad_s": (0, 0.01, 0)}, "angular_speed"),
    ({"table_contact": False}, "table_contact"),
    ({"within_table": False}, "within_table"),
    ({"bottom_gap_m": 0.0011}, "bottom_height"),
    ({"bottom_gap_m": -0.0011}, "bottom_height"),
    ({"top_face": None, "tilt_deg": 20}, "top_face_determined"),
    ({"position_m": (float("nan"), 0, 0)}, "finite_values"),
    ({"linear_velocity_m_s": (float("inf"), 0, 0)}, "finite_values"),
])
def test_window_checks_every_step(settings, settled, change, failed_check):
    samples = list(settled.samples)
    # 最終フレームではなく、検証ウィンドウ先頭だけを不正にする。
    index = settings.steps - settings.window_steps
    samples[index] = replace(samples[index], **change)
    validation = validate_result(replace(settled, samples=samples), settings)
    assert not validation["passed"]
    assert failed_check in validation["failure_reasons"]


def test_early_falling_is_not_settle_failure(settings, settled):
    assert math.hypot(*settled.samples[10].linear_velocity_m_s) > 0.001
    assert validate_result(settled, settings)["passed"]


@pytest.mark.parametrize("kind", ["count", "time", "warning", "empty"])
def test_invalid_run_metadata(settings, settled, kind):
    if kind == "count":
        result = replace(settled, samples=settled.samples[:-1])
        expected = "sample_count"
    elif kind == "time":
        result = replace(settled, samples=[
            replace(s, time_s=s.time_s + 0.1) for s in settled.samples
        ])
        expected = "simulation_times"
    elif kind == "warning":
        result = replace(settled, warnings={"mjWARN_BADQACC": 1})
        expected = "no_mujoco_warnings"
    else:
        result = replace(settled, samples=[])
        expected = "sample_count"
    validation = validate_result(result, settings)
    assert not validation["passed"]
    assert expected in validation["failure_reasons"]


def test_initial_penetration_rejected(settings):
    with pytest.raises(ValueError, match="初期状態"):
        run_simulation(replace(settings, initial_position_m=(0, 0, 0.4)))


def test_fixed_run_is_repeatable(settings, settled):
    repeated = run_simulation(settings)
    np.testing.assert_allclose(
        repeated.samples[-1].position_m, settled.samples[-1].position_m,
        rtol=0, atol=1e-12,
    )
    assert repeated.samples[-1].top_face == settled.samples[-1].top_face
