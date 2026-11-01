import mujoco
import numpy as np

from .placement import PLACE_INITIAL_POSITION_M, placement_grasp_pose


POSITION_OFFSET_LIMIT_M = .005


def validate_position_offset(offset: tuple[float, float]) -> np.ndarray:
    value = np.asarray(offset, dtype=float)
    if (
        value.shape != (2,) or not np.isfinite(value).all()
        or np.any(np.abs(value) > POSITION_OFFSET_LIMIT_M + 1e-12)
    ):
        raise ValueError("初期位置ずれはX・Yそれぞれ±0.005 m以内の有限値にしてください")
    return value


def plan_observed_grasp(
    model: mujoco.MjModel, data: mujoco.MjData, samples: list[dict], face: int,
) -> tuple[np.ndarray, dict]:
    settled = [s for s in samples if 1 - 1e-9 <= s["time_s"] <= 2 + 1e-9]
    if len(settled) != 501 or not all(
        s["dice_table_contact"] and s["dice_table_force_n"] > .01
        and abs(s["dice_bottom_gap_m"]) <= .001
        and s["dice_linear_speed_m_s"] < .001
        and s["dice_angular_speed_rad_s"] < .01
        and s["unexpected_contacts"] == 0 for s in settled
    ):
        raise ValueError("観測計画前の天板支持・静止を確認できません")
    dice = model.body("dice").id
    center = data.xpos[dice].copy()
    rotation = data.xmat[dice].reshape(3, 3)
    if not np.allclose(rotation, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("観測計画は初期上面1・方位固定だけに対応します")
    offset = validate_position_offset(tuple(center[:2] - np.array(PLACE_INITIAL_POSITION_M[:2])))
    base_xy, roll = placement_grasp_pose(face)
    xy = base_xy + offset
    return xy, {
        "mode": "observed_position", "observation_time_s": float(data.time),
        "observed_dice_center_m": center.tolist(),
        "observed_dice_quaternion_wxyz": data.xquat[dice].tolist(),
        "observed_xy_offset_m": offset.tolist(), "support_window_samples": len(settled),
        "grasp_reference_xy_m": xy.tolist(), "wrist_roll_rad": roll,
        "policy": "観測XYに面別の既知の把持オフセットを加えIKを解く。経路順序は面別テンプレート。",
    }
