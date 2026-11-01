import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .approach import OPEN_ANGLE_RAD, TIMESTEP_S, solve_approach


PLACE_INITIAL_POSITION_M = (0.245, 0.015, 0.016)
PLACE_WRIST_ROLL_RAD = -1.52
PLACE_GRIPPER_TORQUE_NM = 0.25
PLACE_X_M = 0.36
PLACE_END_S = 42.0
FINAL_START_S = 41.0
SUPPORT_START_S = 30.5
SUPPORT_END_S = 31.0
MIN_RETRACT_CLEARANCE_M = 0.02
SUPPORTED_PLACEMENT_FACES = (1, 2, 3)


@dataclass(frozen=True)
class PlacementTiming:
    grasp_hold_start_s: float
    grasp_end_s: float
    rotation_hold_start_s: float
    rotation_end_s: float
    support_start_s: float
    support_end_s: float
    final_start_s: float
    end_s: float


def placement_timing(face: int, *, observe_position: bool = False) -> PlacementTiming:
    require_supported_face(face)
    delay = 3.0 if observe_position else 0.0
    offset = (4.0 if face == 3 else 0.0) + delay
    return PlacementTiming(
        14.0 + delay, 16.0 + delay,
        20.0 + offset, 22.0 + offset, SUPPORT_START_S + offset,
        SUPPORT_END_S + offset, FINAL_START_S + offset, PLACE_END_S + offset,
    )


def placement_grasp_pose(face: int) -> tuple[np.ndarray, float]:
    require_supported_face(face)
    return (np.array([.26, .02]), 0.0) if face == 3 else (
        np.array([.25, 0.0]), PLACE_WRIST_ROLL_RAD,
    )


def require_supported_face(face: int) -> None:
    if type(face) is not int or face not in SUPPORTED_PLACEMENT_FACES:
        raise ValueError("配置検証の対応目標面は1・2・3です。他の面は調査段階です")


def placement_target_x(face: int) -> float:
    require_supported_face(face)
    return PLACE_INITIAL_POSITION_M[0] if face == 1 else PLACE_X_M


def plan_placement(
    model: mujoco.MjModel, data: mujoco.MjData, *, target_face: int = 2,
    return_grasp_xy: np.ndarray | None = None,
    observe_position: bool = False,
) -> tuple[list[tuple[str, np.ndarray, float]], dict]:
    require_supported_face(target_face)
    # 実測した把持位置関係を計画に使うが、実行用dataは変更しない。
    planning = mujoco.MjData(model)
    planning.qpos[:] = data.qpos
    gripper = model.body("gripper").id
    dice = model.body("dice").id
    rotation = data.xmat[gripper].reshape(3, 3).copy()
    relative = rotation.T @ (data.xpos[dice] - data.xpos[gripper])
    extent = np.abs(data.xmat[dice].reshape(3, 3)) @ model.geom_size[model.geom("dice_geom").id]
    seed = data.qpos[:6].copy()
    details = {}

    def solve(name: str, height: float, start: np.ndarray) -> np.ndarray:
        q = start.copy()

        def error(angles: np.ndarray) -> np.ndarray:
            planning.qpos[:6] = angles
            mujoco.mj_forward(model, planning)
            actual_rotation = planning.xmat[gripper].reshape(3, 3)
            predicted = planning.xpos[gripper] + actual_rotation @ relative
            # Yは機構と姿勢から決まる値とし、任意の6次元姿勢を要求しない。
            return np.concatenate((
                np.array([PLACE_X_M, height]) - predicted[[0, 2]],
                0.05 * (rotation - actual_rotation).ravel(),
            ))

        for iteration in range(400):
            residual = error(q)
            jacobian = np.empty((11, 5))
            for column in range(5):
                perturbed = q.copy()
                perturbed[column] += 1e-6
                jacobian[:, column] = (error(perturbed) - residual) / 1e-6
            delta = -jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-5 * np.eye(11), residual
            )
            q[:5] = np.clip(
                q[:5] + np.clip(delta, -0.1, 0.1),
                model.jnt_range[:5, 0] + 0.02,
                model.jnt_range[:5, 1] - 0.02,
            )
            residual = error(q)
            if np.linalg.norm(residual) < 1e-6:
                predicted = planning.xpos[gripper] + planning.xmat[gripper].reshape(3, 3) @ relative
                details[name] = {
                    "iterations": iteration + 1,
                    "residual_norm": float(np.linalg.norm(residual)),
                    "predicted_dice_center_m": predicted.tolist(),
                }
                q[5] = 0.0
                return q
        raise ValueError(f"配置姿勢のIKが収束しません: {name}")

    if target_face == 1:
        xy = np.array([.25, 0.0]) if return_grasp_xy is None else return_grasp_xy
        above, above_details = solve_approach(
            model, np.array([*xy, .065]), wrist_roll=PLACE_WRIST_ROLL_RAD
        )
        touch, touch_details = solve_approach(
            model, np.array([*xy, .020]), wrist_roll=PLACE_WRIST_ROLL_RAD
        )
        above[5] = touch[5] = 0.0
        details = {"above": above_details, "touch": touch_details}
    else:
        above = solve("above", 0.060, seed)
        touch = solve("touch", float(extent[2] - 0.0002), above)
    opened = touch.copy()
    opened[5] = OPEN_ANGLE_RAD
    retreat = above.copy()
    retreat[5] = OPEN_ANGLE_RAD
    phases = [
        ("place_above", above, 4.0),
        ("touch", touch, 5.0),
        ("release", opened, 4.0),
        ("retreat", retreat, 4.0),
        ("final_settle", retreat, 3.0),
    ]
    for _, target, _ in phases:
        if np.any(target < model.actuator_ctrlrange[:, 0]) or np.any(
            target > model.actuator_ctrlrange[:, 1]
        ):
            raise ValueError("配置計画の指令が関節範囲外です")
    return phases, {
        "target_x_m": (
            placement_target_x(target_face) if target_face != 1 or return_grasp_xy is None
            else float(
                PLACE_INITIAL_POSITION_M[0] + return_grasp_xy[0] - placement_grasp_pose(1)[0][0]
            )
        ),
        "return_grasp_reference_xy_m": (
            None if return_grasp_xy is None else return_grasp_xy.tolist()
        ),
        "requested_face": target_face,
        "timing_s": vars(placement_timing(target_face, observe_position=observe_position)),
        "strategy": (
            "上面1を維持して元の把持位置へ戻す。回転量0、別位置への移送は行わない。"
            if target_face == 1 else "実測把持関係を使い姿勢を維持してX/Z配置"
        ),
        "target_y_policy": "姿勢と機構で決まるYを採用し、Y位置精度の要求は設けない",
        "measured_dice_in_gripper_m": relative.tolist(),
        "touch_compliance_offset_m": None if target_face == 1 else -0.0002,
        "ik": details,
        "joint_targets_rad": {name: target.tolist() for name, target, _ in phases},
    }


def support_ready(
    samples: list[dict], *, target_face: int = 2, observe_position: bool = False,
) -> bool:
    timing = placement_timing(target_face, observe_position=observe_position)
    window = [
        s for s in samples
        if timing.support_start_s - 1e-9 <= s["time_s"] <= timing.support_end_s + 1e-9
    ]
    return stable_support_window(window, target_face=target_face)


def stable_support_window(window: list[dict], *, target_face: int = 2) -> bool:
    return len(window) == 251 and all(
        s["dice_table_contact"] and s["dice_table_force_n"] > 0.01
        and abs(s["dice_bottom_gap_m"]) <= 0.001
        and s["dice_linear_speed_m_s"] < 0.001
        and s["dice_angular_speed_rad_s"] < 0.01
        and s["top_face"] == target_face and 0 <= s["tilt_deg"] <= 5
        for s in window
    )


def validate_placement(
    samples: list[dict], warnings: dict, *, target_face: int = 2,
    observe_position: bool = False,
    target_x_m: float | None = None,
) -> dict:
    timing = placement_timing(target_face, observe_position=observe_position)
    final = [s for s in samples if s["phase"] == "final_evaluation"]
    if not final:
        raise ValueError("配置後の静止評価サンプルがありません")
    checks = {
        "finite_values": all(
            math.isfinite(value) for s in samples for value in s.values()
            if isinstance(value, (int, float))
        ),
        "sample_count": len(samples) == round(timing.end_s / TIMESTEP_S) + 1,
        "simulation_times": all(
            math.isclose(s["time_s"], i * TIMESTEP_S, abs_tol=1e-9, rel_tol=0)
            for i, s in enumerate(samples)
        ),
        "full_final_window": len(final) == 501 and all(
            (s["phase"] == "final_evaluation") == (i * TIMESTEP_S >= timing.final_start_s)
            for i, s in enumerate(samples)
        ),
        "supported_before_release": support_ready(
            samples, target_face=target_face, observe_position=observe_position,
        ),
        "no_mujoco_warnings": not warnings,
        "no_unexpected_contacts": all(s["unexpected_contacts"] == 0 for s in samples),
        "joint_and_force_limits": all(s["limits_ok"] for s in samples),
        "joint_margin": all(s["min_arm_joint_margin_rad"] >= 0.1 for s in samples),
        "requested_face_up": all(
            s["top_face"] == target_face and 0 <= s["tilt_deg"] <= 5 for s in final
        ),
        "supported_on_table": all(
            s["dice_table_contact"] and s["dice_table_force_n"] > 0.01
            and s["dice_within_table"] and abs(s["dice_bottom_gap_m"]) <= 0.001
            for s in final
        ),
        "stationary": all(
            s["dice_linear_speed_m_s"] < 0.001
            and s["dice_angular_speed_rad_s"] < 0.01 for s in final
        ),
        "no_robot_contact": all(s["dice_robot_contact_count"] == 0 for s in final),
        "gripper_open": all(abs(s["gripper_qpos"] - OPEN_ANGLE_RAD) < 0.01 for s in final),
        "robot_retracted": all(
            isinstance(s["robot_clearance_m"], (int, float))
            and math.isfinite(s["robot_clearance_m"])
            and s["robot_clearance_m"] >= MIN_RETRACT_CLEARANCE_M for s in final
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_reasons": [key for key, passed in checks.items() if not passed],
        "requested_face": target_face, "final_window_start_s": timing.final_start_s,
        "final_window_end_s": timing.end_s, "final_window_samples": len(final),
        "min_robot_clearance_m": min(
            (s["robot_clearance_m"] for s in final
             if isinstance(s["robot_clearance_m"], (int, float))), default=None,
        ),
        "required_robot_clearance_m": MIN_RETRACT_CLEARANCE_M,
        "max_tilt_deg": max(s["tilt_deg"] for s in final),
        "max_linear_speed_m_s": max(s["dice_linear_speed_m_s"] for s in final),
        "max_angular_speed_rad_s": max(s["dice_angular_speed_rad_s"] for s in final),
        "max_abs_bottom_gap_m": max(abs(s["dice_bottom_gap_m"]) for s in final),
        "final_x_error_m": final[-1]["dice_x_m"] - (
            placement_target_x(target_face) if target_x_m is None else target_x_m
        ),
        "position_accuracy_threshold": None,
    }
