import numpy as np
import pytest

from so101_dice_sim.offline_diagnostics import (
    _action_metrics,
    _aligned_sample_time,
    _quat_angle_rad,
    summarize_errors,
    _validate_action_chunk,
)


def test_summarize_errors_reports_joint_and_global_metrics():
    errors = np.array([[1.0, -2.0, 0, 0, 0, 0], [-1.0, 2.0, 0, 0, 0, 0]])
    result = summarize_errors(errors)

    assert result["sample_count"] == 2
    assert result["scalar_count"] == 12
    assert result["mae_rad"] == pytest.approx(.5)
    assert result["rmse_rad"] == pytest.approx(np.sqrt(5 / 6))
    assert result["max_abs_rad"] == 2.0
    assert result["per_joint"]["shoulder_pan"]["mae_rad"] == 1.0


def test_summarize_errors_rejects_invalid_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="6関節"):
        summarize_errors(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="非有限値"):
        summarize_errors(np.full((1, 6), np.nan))


def test_action_metrics_excludes_padded_chunk_targets():
    errors = np.zeros((2, 10, 6))
    errors[0, :, 0] = .1
    errors[1, :, 0] = 9.0
    pads = np.zeros((2, 10), dtype=bool)
    pads[1, 2:] = True
    applied_errors = errors[:, 0, :]
    corrections = np.zeros((2, 6))

    result = _action_metrics(
        errors, pads, np.array([0, 1]), applied_errors, corrections,
    )

    assert result["first_action_raw"]["sample_count"] == 2
    assert result["whole_valid_chunk"]["sample_count"] == 12
    assert result["whole_valid_chunk"]["per_joint"]["shoulder_pan"]["mae_rad"] == pytest.approx(1.5833333)


def test_quaternion_error_handles_sign_equivalence_and_orientation_difference():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_angle_rad(identity, -identity) == pytest.approx(0.0)
    assert _quat_angle_rad(identity, np.array([0.0, 1.0, 0.0, 0.0])) == pytest.approx(np.pi)


def test_dataset_time_alignment_allows_float32_timestamp_rounding():
    assert _aligned_sample_time(float(np.float32(32.6)), 326) == pytest.approx(32.6)
    with pytest.raises(ValueError, match="timestamp"):
        _aligned_sample_time(32.65, 326)


def test_action_chunk_is_verified_against_teacher_endpoints_and_padding():
    endpoints = np.arange(18, dtype=np.float32).reshape(3, 6)
    labels = np.vstack((endpoints[1], endpoints[2], endpoints[2]))
    valid, padded = _validate_action_chunk(
        labels, np.array([False, False, True]), endpoints, frame_index=1,
    )
    assert (valid, padded) == (2, 1)

    labels[0, 0] += 1
    with pytest.raises(ValueError, match="教師次周期末action"):
        _validate_action_chunk(labels, np.array([False, False, True]), endpoints, frame_index=1)


def test_action_chunk_rejects_padding_inside_the_teacher_episode():
    endpoints = np.zeros((3, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="padding扱い"):
        _validate_action_chunk(
            np.zeros((2, 6), dtype=np.float32),
            np.array([False, True]),
            endpoints,
            frame_index=1,
        )
