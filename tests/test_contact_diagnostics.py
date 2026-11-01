import mujoco
import numpy as np
import pytest

from so101_dice_sim.contact_diagnostics import (
    _clearance_summary,
    _measure_clearance,
    _policy_pose,
    _reported_contact_pairs,
    _time_axis,
)
from so101_dice_sim.robot_check import JOINT_NAMES


def _geometry_model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom name="table" type="box" pos="0 0 -0.1" size="1 1 0.1"/>
        <body name="fixed_jaw" pos="0 0 0.06">
          <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0.2"/>
          <geom name="fixed_jaw_box3" type="box" size="0.05 0.05 0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """)


def test_time_axis_accepts_exact_physics_steps_and_rejects_misalignment():
    _time_axis(np.array([0.0, 0.002, 0.004]), 0.002, 0.004)

    with pytest.raises(ValueError, match="時間軸"):
        _time_axis(np.array([0.0, 0.002, 0.005]), 0.002, 0.005)


def test_policy_pose_requires_finite_joints_and_unit_quaternion():
    row = {
        **{f"{name}_qpos": "0" for name in JOINT_NAMES},
        "dice_x_m": "0.1", "dice_y_m": "0.2", "dice_z_m": "0.3",
        "dice_quat_w": "1", "dice_quat_x": "0",
        "dice_quat_y": "0", "dice_quat_z": "0",
    }
    assert _policy_pose(row).shape == (13,)

    row["dice_quat_w"] = "2"
    with pytest.raises(ValueError, match="単位長"):
        _policy_pose(row)
    del row["wrist_roll_qpos"]
    with pytest.raises(ValueError, match="列または値"):
        _policy_pose(row)


def test_reported_contact_pairs_parses_and_rejects_malformed_pairs():
    assert _reported_contact_pairs(["unexpected_contacts:[(0, 32)]"]) == [(0, 32)]

    with pytest.raises(ValueError, match="接触ペア"):
        _reported_contact_pairs(["unexpected_contacts:not-a-pair"])


def test_geom_clearance_matches_known_gap_and_penetration_without_physics_step(monkeypatch):
    model = _geometry_model()
    data = mujoco.MjData(model)
    table = model.geom("table").id
    jaw = model.geom("fixed_jaw_box3").id

    def forbidden_step(*args, **kwargs):
        raise AssertionError("診断でmj_stepを呼び出してはいけません")

    monkeypatch.setattr(mujoco, "mj_step", forbidden_step)

    gap, gap_contact = _measure_clearance(model, data, np.array([0.0]), table, jaw)
    overlap, overlap_contact = _measure_clearance(
        model, data, np.array([-0.02]), table, jaw,
    )

    assert gap == pytest.approx(0.01, abs=1e-8)
    assert gap_contact is False
    assert overlap == pytest.approx(-0.01, abs=1e-8)
    assert overlap_contact is True


def test_clearance_summary_reports_minimum_and_first_penetration():
    result = _clearance_summary(
        np.array([0.0, 0.002, 0.004]),
        np.array([0.03, 0.0, -0.01]),
    )

    assert result["minimum_distance_m"] == pytest.approx(-0.01)
    assert result["minimum_distance_time_s"] == pytest.approx(0.004)
    assert result["first_nonpositive_distance_time_s"] == pytest.approx(0.002)
