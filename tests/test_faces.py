import math

import mujoco
import numpy as np
import pytest

from so101_dice_sim.faces import FACE_AXES, identify_top_face


def rotation(axis, angle_deg):
    radians = math.radians(angle_deg)
    quaternion = np.array([
        math.cos(radians / 2),
        *(math.sin(radians / 2) * np.asarray(axis)),
    ])
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


@pytest.mark.parametrize("axis,angle,expected", [
    ((1, 0, 0), 0, 1),
    ((1, 0, 0), 180, 6),
    ((0, 1, 0), -90, 2),
    ((0, 1, 0), 90, 5),
    ((1, 0, 0), 90, 3),
    ((1, 0, 0), -90, 4),
    ((0, 0, 1), 73, 1),
])
def test_all_faces_and_yaw(axis, angle, expected):
    face, tilt = identify_top_face(rotation(axis, angle), 5)
    assert face == expected
    assert tilt == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("angle,expected", [
    (4.999, 1), (5, 1), (5.001, None), (30, None), (45, None),
])
def test_face_tolerance_boundary(angle, expected):
    face, tilt = identify_top_face(rotation((1, 0, 0), angle), 5)
    assert face == expected
    assert tilt == pytest.approx(angle)


def test_face_map():
    assert set(FACE_AXES) == set(range(1, 7))
    for face, axis in FACE_AXES.items():
        np.testing.assert_array_equal(
            np.asarray(axis), -np.asarray(FACE_AXES[7 - face])
        )


@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)), np.full((3, 3), np.nan),
    np.diag([1.0, 1.0, -1.0]), np.ones((2, 2)),
])
def test_invalid_rotation(matrix):
    with pytest.raises(ValueError):
        identify_top_face(matrix, 5)
