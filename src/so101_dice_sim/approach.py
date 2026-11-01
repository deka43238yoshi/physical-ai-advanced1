import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from .faces import identify_top_face
from .reporting import write_json
from .robot_check import JOINT_NAMES, ROOT, verify_assets


POSITION_TOLERANCE_M = 0.002
AXIS_TOLERANCE_DEG = 3.0
OPEN_ANGLE_RAD = 0.6
DURATION_S = 8.0
TIMESTEP_S = 0.002
EVALUATION_START_S = 7.0


def tool_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    fixed = model.geom("fixed_jaw_sph_tip1").id
    moving = model.geom("moving_jaw_sph_tip1").id
    site = model.site("gripperframe").id
    return (
        (data.geom_xpos[fixed] + data.geom_xpos[moving]) / 2,
        data.site_xmat[site].reshape(3, 3)[:, 0].copy(),
    )


def solve_approach(
    model: mujoco.MjModel, target: np.ndarray, *, wrist_roll: float | None = None,
) -> tuple[np.ndarray, dict]:
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("到達目標は有限の3次元位置にしてください")
    # IK用の独立データのみを書き換える。実行中の物理状態には触れない。
    planning = mujoco.MjData(model)
    q = np.array([0.0, -0.3, 0.7, 0.7, 0.0, OPEN_ANGLE_RAD])
    if wrist_roll is not None:
        if not math.isfinite(wrist_roll) or not (
            model.jnt_range[4, 0] + 0.01 <= wrist_roll <= model.jnt_range[4, 1] - 0.01
        ):
            raise ValueError("手首回転の指定が関節範囲外です")
        q[4] = wrist_roll
    down = np.array([0.0, 0.0, -1.0])

    def residual(angles: np.ndarray) -> np.ndarray:
        planning.qpos[:6] = angles
        mujoco.mj_forward(model, planning)
        position, axis = tool_pose(model, planning)
        error = np.concatenate((target - position, 0.1 * (down - axis)))
        if wrist_roll is not None:
            error = np.append(error, 0.1 * (wrist_roll - angles[4]))
        return error

    for iteration in range(300):
        error = residual(q)
        jacobian = np.empty((len(error), 5))
        for column in range(5):
            perturbed = q.copy()
            perturbed[column] += 1e-6
            jacobian[:, column] = (residual(perturbed) - error) / 1e-6
        delta = -jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-5 * np.eye(len(error)), error
        )
        q[:5] = np.clip(
            q[:5] + np.clip(delta, -0.1, 0.1),
            model.jnt_range[:5, 0] + 0.01,
            model.jnt_range[:5, 1] - 0.01,
        )
        error = residual(q)
        position_error = float(np.linalg.norm(error[:3]))
        _, axis = tool_pose(model, planning)
        angle_error = math.degrees(math.acos(float(np.clip(axis @ down, -1, 1))))
        roll_ok = wrist_roll is None or abs(q[4] - wrist_roll) < 0.001
        if position_error < 0.0001 and angle_error < 0.1 and roll_ok:
            return q, {
                "iterations": iteration + 1,
                "position_error_m": position_error,
                "axis_error_deg": angle_error,
                "target_position_m": target.tolist(),
                "target_axis_world": down.tolist(),
                "target_joint_angles_rad": q.tolist(),
            }
    raise ValueError(
        f"到達姿勢のIKが収束しません: 位置誤差={position_error} m, "
        f"方向誤差={angle_error} deg"
    )


def validate_approach(samples: list[dict], warnings: dict) -> dict:
    if not samples or not any(s["evaluation_window"] for s in samples):
        raise ValueError("接近検証の評価サンプルがありません")
    window = [s for s in samples if s["evaluation_window"]]
    numeric = [
        value for s in samples for key, value in s.items()
        if key not in ("phase", "top_face") and isinstance(value, (float, int))
    ]
    checks = {
        "finite_values": all(math.isfinite(v) for v in numeric),
        "sample_count": len(samples) == round(DURATION_S / TIMESTEP_S) + 1,
        "simulation_times": all(
            math.isclose(s["time_s"], i * TIMESTEP_S, abs_tol=1e-9, rel_tol=0)
            for i, s in enumerate(samples)
        ),
        "evaluation_window": len(window) == 501 and all(
            s["evaluation_window"] == (i * TIMESTEP_S >= EVALUATION_START_S)
            for i, s in enumerate(samples)
        ),
        "no_mujoco_warnings": not warnings,
        "no_unexpected_contacts": all(s["unexpected_contacts"] == 0 for s in samples),
        "joint_and_force_limits": all(s["limits_ok"] for s in samples),
        "reached_position": all(
            s["position_error_m"] < POSITION_TOLERANCE_M for s in window
        ),
        "downward_tool_axis": all(
            s["axis_error_deg"] < AXIS_TOLERANCE_DEG for s in window
        ),
        "arm_settled": all(s["max_joint_speed_rad_s"] < 0.01 for s in window),
        "dice_on_table": all(
            s["dice_table_contact"] and s["dice_within_table"]
            and abs(s["dice_bottom_gap_m"]) <= 0.001 for s in window
        ),
        "dice_settled": all(
            s["dice_linear_speed_m_s"] < 0.001
            and s["dice_angular_speed_rad_s"] < 0.01 for s in window
        ),
        "dice_not_displaced": all(s["dice_xy_displacement_m"] < 0.001 for s in samples),
        "dice_top_face": all(s["top_face"] == 1 for s in window),
        "gripper_remains_open": all(
            abs(s["gripper_qpos"] - OPEN_ANGLE_RAD) < 0.01 for s in samples
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_reasons": [k for k, v in checks.items() if not v],
        "max_position_error_m": max(s["position_error_m"] for s in window),
        "max_axis_error_deg": max(s["axis_error_deg"] for s in window),
        "max_joint_speed_rad_s": max(s["max_joint_speed_rad_s"] for s in window),
        "unexpected_contact_steps": sum(s["unexpected_contacts"] > 0 for s in samples),
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "axis_tolerance_deg": AXIS_TOLERANCE_DEG,
    }


def run_approach(directory: Path) -> dict:
    provenance = verify_assets(ROOT / "assets" / "robotstudio_so101")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    if (model.nq, model.nv, model.nu) != (13, 12, 6):
        raise ValueError("統合シーンの関節構成が想定と一致しません")
    for index, name in enumerate(JOINT_NAMES):
        if (
            model.joint(name).id != index
            or model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE
            or model.actuator_trnid[index, 0] != index
        ):
            raise ValueError(f"アーム関節の順序が想定と異なります: {name}")
    dice = model.body("dice").id
    dice_geom = model.geom("dice_geom").id
    table = model.geom("table").id
    initial_xy = model.body_pos[dice, :2].copy()
    target = np.array([*initial_xy, 0.085])
    solution, planning = solve_approach(model, target)
    data = mujoco.MjData(model)
    data.qpos[5] = OPEN_ANGLE_RAD
    data.ctrl[:] = data.qpos[:6]
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy().tolist()
    unexpected_pairs: set[tuple[int, int]] = set()
    samples = []

    def observe(phase: str, evaluate: bool) -> None:
        if not all(np.isfinite(a).all() for a in (
            data.qpos, data.qvel, data.qacc, data.actuator_force,
        )):
            raise FloatingPointError("統合シーンの状態に非有限値が発生しました")
        position, axis = tool_pose(model, data)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, dice, velocity, 0
        )
        rotation = data.xmat[dice].reshape(3, 3)
        extent = np.abs(rotation) @ model.geom_size[dice_geom]
        center = data.xpos[dice]
        table_center = data.geom_xpos[table]
        table_half = model.geom_size[table]
        face, _ = identify_top_face(rotation, 5)
        unexpected = 0
        table_contact = False
        for contact in data.contact:
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if set(pair) == {dice_geom, table}:
                table_contact = True
            else:
                unexpected += 1
                unexpected_pairs.add(pair)
        row = {
            "time_s": float(data.time), "phase": phase,
            "evaluation_window": evaluate,
            **{f"tool_{a}_m": float(v) for a, v in zip("xyz", position)},
            "position_error_m": float(np.linalg.norm(position - target)),
            "axis_error_deg": math.degrees(math.acos(float(np.clip(-axis[2], -1, 1)))),
            "max_joint_speed_rad_s": float(np.max(np.abs(data.qvel[:6]))),
            "unexpected_contacts": unexpected,
            "dice_table_contact": table_contact,
            "dice_bottom_gap_m": float(
                center[2] - extent[2] - table_center[2] - table_half[2]
            ),
            "dice_within_table": bool(np.all(
                np.abs(center[:2] - table_center[:2]) + extent[:2] <= table_half[:2]
            )),
            "dice_xy_displacement_m": float(np.linalg.norm(center[:2] - initial_xy)),
            "dice_linear_speed_m_s": float(np.linalg.norm(velocity[3:])),
            "dice_angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
            "top_face": face,
            **{f"dice_{a}_m": float(v) for a, v in zip("xyz", center)},
            **{f"dice_quat_{a}": float(v) for a, v in zip("wxyz", data.xquat[dice])},
            "limits_ok": bool(
                np.all(data.qpos[:6] >= model.jnt_range[:6, 0] - 1e-6)
                and np.all(data.qpos[:6] <= model.jnt_range[:6, 1] + 1e-6)
                and np.all(data.actuator_force >= model.actuator_forcerange[:, 0] - 1e-6)
                and np.all(data.actuator_force <= model.actuator_forcerange[:, 1] + 1e-6)
            ),
        }
        for i, name in enumerate(JOINT_NAMES):
            row.update({
                f"{name}_qpos": float(data.qpos[i]),
                f"{name}_qvel": float(data.qvel[i]),
                f"{name}_target": float(data.ctrl[i]),
                f"{name}_force": float(data.actuator_force[i]),
            })
        samples.append(row)

    observe("initial", False)
    start = data.qpos[:6].copy()
    for step in range(round(DURATION_S / TIMESTEP_S)):
        time = (step + 1) * model.opt.timestep
        fraction = float(np.clip((time - 2.0) / 3.0, 0, 1))
        smooth = fraction * fraction * (3 - 2 * fraction)
        data.ctrl[:] = start + smooth * (solution - start)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        observe(
            "settle" if time <= 2 else "approach" if time <= 5 else "hold",
            time >= EVALUATION_START_S,
        )
    warnings = {
        mujoco.mjtWarning(i).name: int(w.number)
        for i, w in enumerate(data.warning) if w.number
    }
    validation = validate_approach(samples, warnings)
    report = {
        "schema_version": 1, "task": "so101_dice_pregrasp_approach",
        "mujoco_version": mujoco.__version__,
        "source_commit": provenance["commit"],
        "units": {
            "length": "m", "time": "s", "angle": "rad",
            "axis_error": "deg", "linear_speed": "m/s", "angular_speed": "rad/s",
            "torque": "N*m", "quaternion": "wxyz", "frame": "world",
        },
        "initial_qpos": initial,
        "table_top_m": 0.0,
        "table_half_size_m": model.geom_size[table].tolist(),
        "dice_edge_m": float(model.geom_size[dice_geom, 0] * 2),
        "dice_mass_kg": float(model.body_mass[dice]),
        "joints": [
            {
                "name": name, "range_rad": model.jnt_range[i].tolist(),
                "force_range_nm": model.actuator_forcerange[i].tolist(),
            }
            for i, name in enumerate(JOINT_NAMES)
        ],
        "planning": planning,
        "duration_s": float(data.time), "sample_count": len(samples),
        "timestep_s": float(model.opt.timestep),
        "evaluation_start_s": EVALUATION_START_S,
        "evaluation_samples": sum(s["evaluation_window"] for s in samples),
        "warnings": warnings,
        "unexpected_contact_pairs": [
            [model.geom(g).name or f"geom-{g}" for g in pair]
            for pair in sorted(unexpected_pairs)
        ],
        "validation": validation,
        "final_state": samples[-1],
        "limitations": [
            "指先2球の中心間中点を把持準備位置と定義。把持点の最適化ではない。",
            "手先の長手方向を下向きにするが、任意の6次元手先姿勢は指定しない。",
            "固定配置・1軌道の接触前到達検証のみ。閉指・把持・持ち上げは未実施。",
            "基部の衝突形状は上流にないため設置面の干渉検証対象外。",
        ],
    }
    with (directory / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    write_json(directory / "result.json", report)
    return report
