import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from .approach import OPEN_ANGLE_RAD, TIMESTEP_S, solve_approach
from .faces import identify_top_face
from .reporting import write_json
from .robot_check import JOINT_NAMES, ROOT, verify_assets
from .rotation import TARGET_FACE, validate_rotation
from .observation import plan_observed_grasp, validate_position_offset
from .placement import (
    PLACE_GRIPPER_TORQUE_NM, PLACE_INITIAL_POSITION_M,
    placement_grasp_pose, placement_timing, plan_placement, require_supported_face,
    support_ready, validate_placement,
)


INITIAL_DICE_POSITION_M = (0.235, 0.0, 0.016)
MIN_CLEARANCE_M = 0.03
MIN_JAW_FORCE_N = 0.01
MAX_HOLD_DRIFT_M = 0.002
DURATION_S = 16.0
HOLD_START_S = 14.0
JOINT_MARGIN_RAD = 0.10
GRIPPER_TORQUE_LIMIT_NM = 0.2


def contact_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[dict, list[tuple[int, int]]]:
    dice = model.geom("dice_geom").id
    table = model.geom("table").id
    fixed_body = model.body("gripper").id
    moving_body = model.body("moving_jaw_so101_v1").id
    forces = {"fixed_jaw_force_n": 0.0, "moving_jaw_force_n": 0.0}
    table_contact = False
    table_force = 0.0
    robot_contacts = 0
    unexpected = []
    for index, contact in enumerate(data.contact):
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        if set(pair) == {dice, table}:
            table_contact = True
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            table_force += max(float(wrench[0]), 0.0)
            continue
        if dice in pair:
            robot_contacts += 1
            other = pair[1] if pair[0] == dice else pair[0]
            body = int(model.geom_bodyid[other])
            if body in (fixed_body, moving_body) and model.geom_group[other] in (3, 4):
                wrench = np.empty(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                name = "fixed_jaw_force_n" if body == fixed_body else "moving_jaw_force_n"
                forces[name] += max(float(wrench[0]), 0.0)
                continue
        unexpected.append(pair)
    return {
        **forces, "dice_table_contact": table_contact,
        "dice_table_force_n": table_force,
        "dice_robot_contact_count": robot_contacts,
        "unexpected_contacts": len(unexpected),
    }, unexpected


def capture_sample(
    model: mujoco.MjModel, data: mujoco.MjData, phase: str, *,
    robot_geoms: list[int] | None = None, distance_cap_m: float = 1.0,
) -> tuple[dict, list[tuple[int, int]]]:
    if not all(np.isfinite(v).all() for v in (
        data.qpos, data.qvel, data.qacc, data.actuator_force,
    )):
        raise FloatingPointError("把持シミュレーションに非有限値が発生しました")
    dice = model.body("dice").id
    geom = model.geom("dice_geom").id
    gripper = model.body("gripper").id
    table = model.geom("table").id
    center = data.xpos[dice]
    rotation = data.xmat[dice].reshape(3, 3)
    extent = np.abs(rotation) @ model.geom_size[geom]
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, dice, velocity, 0)
    relative = data.xmat[gripper].reshape(3, 3).T @ (center - data.xpos[gripper])
    contact, unexpected = contact_state(model, data)
    face, tilt = identify_top_face(rotation, 5)
    margin = np.minimum(
        data.qpos[:5] - model.jnt_range[:5, 0],
        model.jnt_range[:5, 1] - data.qpos[:5],
    )
    limits = bool(
        np.all(data.qpos[:6] >= model.jnt_range[:6, 0] - 1e-6)
        and np.all(data.qpos[:6] <= model.jnt_range[:6, 1] + 1e-6)
        and np.all(data.actuator_force >= model.actuator_forcerange[:, 0] - 1e-6)
        and np.all(data.actuator_force <= model.actuator_forcerange[:, 1] + 1e-6)
    )
    row = {
        "time_s": float(data.time), "phase": phase, **contact,
        "dice_bottom_gap_m": float(
            center[2] - extent[2] - data.geom_xpos[table, 2] - model.geom_size[table, 2]
        ),
        "dice_linear_speed_m_s": float(np.linalg.norm(velocity[3:])),
        "dice_angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
        "max_arm_speed_rad_s": float(np.max(np.abs(data.qvel[:6]))),
        "min_arm_joint_margin_rad": float(np.min(margin)),
        "limits_ok": limits,
        "top_face": face, "tilt_deg": tilt,
        **{f"dice_{axis}_m": float(v) for axis, v in zip("xyz", center)},
        **{f"dice_quat_{axis}": float(v) for axis, v in zip("wxyz", data.xquat[dice])},
        **{f"dice_in_gripper_{axis}_m": float(v) for axis, v in zip("xyz", relative)},
    }
    if robot_geoms is not None:
        row.update({
            "dice_within_table": bool(np.all(
                np.abs(center[:2] - data.geom_xpos[table, :2]) + extent[:2]
                <= model.geom_size[table, :2]
            )),
            "robot_clearance_m": (
                min(float(mujoco.mj_geomDistance(
                    model, data, geom, other, distance_cap_m, None,
                )) for other in robot_geoms)
                if phase in ("final_settle", "final_evaluation") else None
            ),
        })
    for i, name in enumerate(JOINT_NAMES):
        row.update({
            f"{name}_qpos": float(data.qpos[i]),
            f"{name}_qvel": float(data.qvel[i]),
            f"{name}_target": float(data.ctrl[i]),
            f"{name}_force": float(data.actuator_force[i]),
        })
    return row, unexpected


def validate_grasp(
    samples: list[dict], warnings: dict, *, observe_position: bool = False,
) -> dict:
    timing = placement_timing(2, observe_position=observe_position)
    if not samples or not any(s["phase"] == "hold" for s in samples):
        raise ValueError("把持保持の評価サンプルがありません")
    hold = [s for s in samples if s["phase"] == "hold"]
    closed = [s for s in samples if 10.5 <= s["time_s"] <= 11.0 + 1e-9]
    settled = [s for s in samples if 1.0 <= s["time_s"] <= 2.0 + 1e-9]
    numeric = [
        value for s in samples for value in s.values()
        if isinstance(value, (int, float))
    ]
    relative = np.array([
        [s[f"dice_in_gripper_{axis}_m"] for axis in "xyz"] for s in hold
    ])
    drift = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
    def both(sample: dict) -> bool:
        return (
            sample["fixed_jaw_force_n"] >= MIN_JAW_FORCE_N
            and sample["moving_jaw_force_n"] >= MIN_JAW_FORCE_N
        )
    checks = {
        "finite_values": all(math.isfinite(v) for v in numeric),
        "sample_count": len(samples) == round(timing.grasp_end_s / TIMESTEP_S) + 1,
        "simulation_times": all(
            math.isclose(s["time_s"], i * TIMESTEP_S, abs_tol=1e-9, rel_tol=0)
            for i, s in enumerate(samples)
        ),
        "full_hold_window": len(hold) == 1001 and all(
            (s["phase"] == "hold") == (i * TIMESTEP_S >= timing.grasp_hold_start_s)
            for i, s in enumerate(samples)
        ),
        "no_mujoco_warnings": not warnings,
        "no_unexpected_contacts": all(s["unexpected_contacts"] == 0 for s in samples),
        "joint_margin": all(
            s["min_arm_joint_margin_rad"] >= JOINT_MARGIN_RAD for s in samples
        ),
        "joint_and_force_limits": all(s["limits_ok"] for s in samples),
        "initial_dice_settled": bool(settled) and all(
            s["dice_table_contact"] and abs(s["dice_bottom_gap_m"]) <= 0.001
            and s["dice_linear_speed_m_s"] < 0.001
            and s["dice_angular_speed_rad_s"] < 0.01 for s in settled
        ),
        "closed_on_both_jaws": bool(closed) and all(both(s) for s in closed),
        "both_jaws_during_hold": all(both(s) for s in hold),
        "off_table_during_hold": all(
            not s["dice_table_contact"] and s["dice_bottom_gap_m"] >= MIN_CLEARANCE_M
            for s in hold
        ),
        "lifted_from_initial_height": all(
            s["dice_z_m"] - INITIAL_DICE_POSITION_M[2] >= 0.035 for s in hold
        ),
        "held_without_slip": math.isfinite(drift) and drift < MAX_HOLD_DRIFT_M,
        "hold_speeds": all(
            s["max_arm_speed_rad_s"] < 0.01
            and s["dice_linear_speed_m_s"] < 0.001
            and s["dice_angular_speed_rad_s"] < 0.01 for s in hold
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_reasons": [key for key, passed in checks.items() if not passed],
        "hold_samples": len(hold),
        "hold_start_s": timing.grasp_hold_start_s,
        "hold_duration_s": timing.grasp_end_s - timing.grasp_hold_start_s,
        "min_clearance_m": min(s["dice_bottom_gap_m"] for s in hold),
        "required_clearance_m": MIN_CLEARANCE_M,
        "min_fixed_jaw_force_n": min(s["fixed_jaw_force_n"] for s in hold),
        "min_moving_jaw_force_n": min(s["moving_jaw_force_n"] for s in hold),
        "required_jaw_force_n": MIN_JAW_FORCE_N,
        "max_fixed_jaw_force_n": max(s["fixed_jaw_force_n"] for s in samples),
        "max_moving_jaw_force_n": max(s["moving_jaw_force_n"] for s in samples),
        "max_relative_drift_m": drift,
        "relative_drift_limit_m": MAX_HOLD_DRIFT_M,
        "min_arm_joint_margin_rad": min(s["min_arm_joint_margin_rad"] for s in samples),
        "required_joint_margin_rad": JOINT_MARGIN_RAD,
    }


def run_grasp(
    directory: Path, *, rotate_to_face: int | None = None, place_face: int | None = None,
    observe_position: bool = False, initial_offset_m: tuple[float, float] = (0.0, 0.0),
    grasp_error_m: tuple[float, float] = (0.0, 0.0),
) -> dict:
    offset = validate_position_offset(initial_offset_m)
    grasp_error = validate_position_offset(grasp_error_m)
    if np.any(grasp_error) and not observe_position:
        raise ValueError("把持目標への誤差注入には観測位置計画が必要です")
    if (observe_position or np.any(offset)) and place_face is None:
        raise ValueError("観測位置計画には配置面の指定が必要です")
    if np.any(offset) and not observe_position:
        raise ValueError("初期位置ずれの指定には観測位置計画が必要です")
    if place_face is not None:
        require_supported_face(place_face)
        if rotate_to_face is not None and rotate_to_face != place_face:
            raise ValueError("回転面と配置面が一致しません")
        rotate_to_face = place_face
    if place_face is None and rotate_to_face is not None and (
        isinstance(rotate_to_face, bool) or rotate_to_face != TARGET_FACE
    ):
        raise ValueError("現在の回転検証は固定初期条件から上面2への回転だけに対応します")
    provenance = verify_assets(ROOT / "assets" / "robotstudio_so101")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    if (model.nq, model.nv, model.nu, model.neq) != (13, 12, 6, 0):
        raise ValueError("把持検証は固定基部アームと自由サイコロ、追加拘束なしを想定します")
    for index, name in enumerate(JOINT_NAMES):
        if (
            model.joint(name).id != index
            or model.actuator_trnid[index, 0] != index
            or not model.jnt_limited[index]
            or not model.actuator_forcelimited[index]
        ):
            raise ValueError(f"関節構造が想定と異なります: {name}")
    dice_joint = model.joint("dice_free")
    if dice_joint.qposadr[0] != 6 or dice_joint.type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("サイコロの自由関節の配置が想定と異なります")
    upstream_gripper_torque_range = model.actuator_forcerange[5].copy().tolist()
    # 小さな物体を上流の最大トルクで締め続けないよう、この検証だけ制限する。
    torque_limit = (
        PLACE_GRIPPER_TORQUE_NM if place_face is not None else GRIPPER_TORQUE_LIMIT_NM
    )
    model.actuator_forcerange[5] = [-torque_limit, torque_limit]
    timing = placement_timing(
        2 if place_face is None else place_face, observe_position=observe_position,
    )
    grasp_xy, grasp_roll = (
        (np.array([.25, 0.0]), None) if place_face is None else placement_grasp_pose(place_face)
    )
    planning = {}

    def make_targets(xy: np.ndarray) -> dict[str, np.ndarray]:
        targets = {}
        for name, height in (("pregrasp", .060), ("lower", .020), ("lift", .065)):
            targets[name], planning[name] = solve_approach(
                model, np.array([*xy, height]), wrist_roll=grasp_roll,
            )
        targets["close"] = targets["lower"].copy()
        targets["close"][5] = 0.0
        targets["lift"][5] = 0.0
        if rotate_to_face is not None:
            targets["rotate"] = targets["lift"].copy()
            targets["rotate"][3] += 0.0 if rotate_to_face == 1 else -math.pi / 2
            if place_face == 3:
                targets["wrist_twist"] = targets["lift"].copy()
                targets["wrist_twist"][4] -= math.pi / 2
                targets["rotate"][4] -= math.pi / 2
        for target in targets.values():
            margin = np.minimum(
                target[:5] - model.jnt_range[:5, 0],
                model.jnt_range[:5, 1] - target[:5],
            )
            if np.min(margin) < JOINT_MARGIN_RAD or np.any(
                target < model.actuator_ctrlrange[:, 0]
            ) or np.any(target > model.actuator_ctrlrange[:, 1]):
                raise ValueError("計画軌道の関節余裕または指令範囲が不適切です")
        return targets

    targets = {} if observe_position else make_targets(grasp_xy)
    data = mujoco.MjData(model)
    data.qpos[5] = OPEN_ANGLE_RAD
    data.qpos[6:9] = (
        PLACE_INITIAL_POSITION_M if place_face is not None else INITIAL_DICE_POSITION_M
    )
    data.qpos[6:8] += offset
    data.ctrl[:] = data.qpos[:6]
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.tolist()
    dice = model.body("dice").id
    samples = []
    pairs: set[tuple[int, int]] = set()
    robot_geoms = [
        i for i in range(model.ngeom)
        if model.geom_bodyid[i] not in (0, dice)
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]

    def observe(phase: str) -> None:
        row, unexpected = capture_sample(
            model, data, phase,
            robot_geoms=robot_geoms if place_face is not None else None,
            distance_cap_m=.05 if observe_position else 1.0,
        )
        pairs.update(unexpected)
        samples.append(row)

    observe("initial")
    phases = [
        ("settle", data.qpos[:6].copy(), 2.0),
    ]

    def append_grasp_phases() -> None:
        phases.extend([
            ("pregrasp", targets["pregrasp"], 3.0),
            ("lower", targets["lower"], 3.0),
            ("close", targets["close"], 3.0),
            ("lift", targets["lift"], 3.0),
            ("hold", targets["lift"], 5.0 if observe_position else 2.0),
        ])
        if rotate_to_face is not None:
            if place_face == 3:
                phases.append(("wrist_twist", targets["wrist_twist"], 4.0))
            phases.extend([
                ("rotate", targets["rotate"], 4.0),
                ("rotation_hold", targets["rotate"], 2.0),
            ])

    if not observe_position:
        append_grasp_phases()
    phase_results = []
    placement_plan = None
    observation_plan = None

    def abort_placement(message: str) -> None:
        with (directory / "aborted_trajectory.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        write_json(directory / "placement_abort.json", {
            "status": "aborted", "reason": message,
            "last_time_s": float(data.time), "sample_count": len(samples),
            "release_executed": False, "final_state": samples[-1],
            "observation_plan": observation_plan,
        })
        raise ValueError(message)
    previous = data.qpos[:6].copy()
    steps = 0
    for phase, target, duration in phases:
        if phase == "release" and not support_ready(
            samples, target_face=place_face, observe_position=observe_position,
        ):
            abort_placement("接地・支持・静止を確認できないため、開指を中止しました")
        for step in range(round(duration / TIMESTEP_S)):
            ramp_s = (
                duration - 1.0 if place_face is not None
                else 3.0 if phase == "rotate" else 2.0
            )
            if place_face in (1, 3) and phase == "touch":
                ramp_s = 3.0
            fraction = min((step + 1) * TIMESTEP_S / ramp_s, 1.0)
            smooth = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[:] = previous + smooth * (target - previous)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            steps += 1
            # 評価窓の開始時刻ちょうども保持サンプルに含める。
            elapsed = steps * TIMESTEP_S
            if place_face is not None and elapsed >= timing.final_start_s:
                observation_phase = "final_evaluation"
            elif (
                rotate_to_face is not None
                and timing.rotation_hold_start_s <= elapsed <= timing.rotation_end_s
            ):
                observation_phase = "rotation_hold"
            elif timing.grasp_hold_start_s <= elapsed <= timing.grasp_end_s:
                observation_phase = "hold"
            elif observe_position and phase == "hold":
                observation_phase = "hold_settle"
            else:
                observation_phase = phase
            observe(observation_phase)
        phase_results.append({
            "phase": phase, "end_s": float(data.time),
            "target_rad": target.tolist(), "actual_rad": data.qpos[:6].tolist(),
            "dice_position_m": data.xpos[dice].tolist(),
        })
        previous = target
        if phase == "settle" and observe_position:
            try:
                grasp_xy, observation_plan = plan_observed_grasp(model, data, samples, place_face)
                if np.any(grasp_error):
                    observation_plan["unbiased_grasp_reference_xy_m"] = grasp_xy.tolist()
                    grasp_xy = grasp_xy + grasp_error
                    observation_plan["injected_grasp_error_m"] = grasp_error.tolist()
                    observation_plan["grasp_reference_xy_m"] = grasp_xy.tolist()
                    observation_plan["policy"] += " 検証専用の人工XY誤差を把持目標に加算。"
                observation_plan.update({
                    "robot_distance_query_cap_m": .05, "extra_hold_settle_s": 3.0,
                })
                write_json(directory / "observation_plan.json", observation_plan)
                targets = make_targets(grasp_xy)
            except ValueError as error:
                abort_placement(f"観測位置からの計画を中止: {error}")
            append_grasp_phases()
        if phase == "rotation_hold" and place_face is not None:
            stage_warnings = {
                mujoco.mjtWarning(i).name: int(w.number)
                for i, w in enumerate(data.warning) if w.number
            }
            grasp_ready = validate_grasp(
                samples[:round(timing.grasp_end_s / TIMESTEP_S) + 1], stage_warnings,
                observe_position=observe_position,
            )
            rotated_ready = validate_rotation(
                samples, stage_warnings, target_face=rotate_to_face,
                observe_position=observe_position,
            )
            if not grasp_ready["passed"] or not rotated_ready["passed"]:
                abort_placement(
                    "把持・回転の検証不合格のため配置を中止: "
                    + ", ".join(
                        grasp_ready["failure_reasons"] + rotated_ready["failure_reasons"]
                    )
                )
            try:
                extra_phases, placement_plan = plan_placement(
                    model, data, target_face=place_face,
                    return_grasp_xy=grasp_xy if observe_position else None,
                    observe_position=observe_position,
                )
            except ValueError as error:
                abort_placement(f"配置姿勢の計画を中止: {error}")
            phases.extend(extra_phases)
    warnings = {
        mujoco.mjtWarning(i).name: int(w.number)
        for i, w in enumerate(data.warning) if w.number
    }
    grasp_samples = samples[:round(timing.grasp_end_s / TIMESTEP_S) + 1]
    grasp_validation = validate_grasp(grasp_samples, warnings, observe_position=observe_position)
    validation = grasp_validation
    rotation_validation = None
    if rotate_to_face is not None:
        rotation_validation = validate_rotation(
            samples[:round(timing.rotation_end_s / TIMESTEP_S) + 1],
            warnings, target_face=rotate_to_face,
            observe_position=observe_position,
        )
        combined_checks = {
            "grasp_and_lift": grasp_validation["passed"],
            **rotation_validation["checks"],
        }
        validation = {
            **rotation_validation,
            "passed": all(combined_checks.values()),
            "checks": combined_checks,
            "failure_reasons": [
                key for key, passed in combined_checks.items() if not passed
            ],
        }
    report = {
        "schema_version": 1,
        "task": (
            "so101_fixed_grasp_rotate" if rotate_to_face is not None
            else "so101_fixed_grasp_lift"
        ),
        "source_commit": provenance["commit"],
        "mujoco_version": mujoco.__version__,
        "units": {
            "length": "m", "time": "s", "angle": "rad", "tilt": "deg",
            "linear_speed": "m/s", "angular_speed": "rad/s",
            "contact_force": "N", "joint_torque": "N*m", "quaternion": "wxyz",
        },
        "duration_s": float(data.time), "timestep_s": TIMESTEP_S,
        "sample_count": len(samples), "equality_constraints": model.neq,
        "initial_qpos": initial_qpos, "planning": planning,
        "runtime_overrides": {
            "gripper_force_range_nm": model.actuator_forcerange[5].tolist(),
            "upstream_gripper_force_range_nm": upstream_gripper_torque_range,
            "reason": "小物把持の過大な締め付けを避けるためトルク上限を低減",
        },
        "phases": phase_results,
        "joints": [
            {"name": name, "range_rad": model.jnt_range[i].tolist(),
             "force_range_nm": model.actuator_forcerange[i].tolist()}
            for i, name in enumerate(JOINT_NAMES)
        ],
        "unexpected_contact_pairs": [
            [model.geom(g).name or f"geom-{g}" for g in pair] for pair in sorted(pairs)
        ],
        "warnings": warnings, "validation": validation,
        "final_state": samples[-1],
        "limitations": [
            "固定位置・姿勢・摩擦の1条件のみ。ランダム条件の把持成功は未検証。",
            "下降時の指との軽い接触と閉指時の物体移動を許容する。",
            "2秒保持の位置滑りを評価。長時間保持や姿勢保持の一般保証ではない。",
            "回転・指定面への配置・解放は未実施。基部の衝突形状は上流にない。",
        ],
    }
    if rotate_to_face is not None:
        report.update({
            "requested_face": rotate_to_face,
            "grasp_validation": grasp_validation,
            "rotation_validation": rotation_validation,
            "rotation_plan": {
                "joint": "wrist_flex", "delta_rad": 0.0 if rotate_to_face == 1 else -math.pi / 2,
                "start_s": timing.grasp_end_s + (4.0 if place_face == 3 else 0.0),
                "ramp_duration_s": 3.0, "hold_start_s": timing.rotation_hold_start_s,
                **({"preceding_wrist_twist": {
                    "joint": "wrist_roll", "delta_rad": -math.pi / 2,
                    "start_s": timing.grasp_end_s, "ramp_duration_s": 3.0,
                    "end_s": timing.grasp_end_s + 4.0,
                }} if place_face == 3 else {}),
            },
            "limitations": [
                "固定初期条件の上面1から2への回転のみ。他の指定面は未対応。",
                "2秒保持のみ。位置滑りは検査するが相対姿勢固定は要求しない。",
                "テーブルへの配置・解放は未実施。最終目的の完了ではない。",
                "基部の衝突形状は上流にない。実機・ランダム条件は未検証。",
            ],
        })
    if place_face is not None:
        placement_validation = validate_placement(
            samples, warnings, target_face=place_face, observe_position=observe_position,
            target_x_m=placement_plan["target_x_m"],
        )
        checks = {
            "grasp_and_lift": grasp_validation["passed"],
            "rotation": rotation_validation["passed"],
            **placement_validation["checks"],
        }
        report.update({
            "task": f"so101_fixed_place_face_{place_face}",
            "placement_plan": placement_plan,
            "placement_validation": placement_validation,
            "validation": {
                **placement_validation, "passed": all(checks.values()),
                "checks": checks,
                "failure_reasons": [key for key, passed in checks.items() if not passed],
            },
            "limitations": [
                f"初期姿勢1上から目標{place_face}上へ置く固定条件のみ。対応面は1・2・3。",
                "配置用の把持方向・指掛かりを使用。面3は手首ロール0からねじりと曲げを実行。",
                "面1は元の把持位置へ戻す。面2・3はX=0.36 mへ移送。Yは機構任せ。位置精度閾値なし。",
                "基部の衝突形状は上流にない。実機・長時間保持は未検証。",
            ],
        })
        if observe_position:
            report["observation_plan"] = observation_plan
            report["initial_xy_offset_m"] = offset.tolist()
            report["planning_mode"] = "observed_position"
            report["limitations"].append(
                "観測位置はXY各±5 mm・初期姿勢固定のみ。観測XYで把持目標を再計算するが面別手順は固定。"
            )
            report["limitations"][0] = (
                f"初期姿勢1上・XY各±5 mmから目標{place_face}への観測位置計画。"
                "検証した有限個の位置以外の成功保証ではない。"
            )
    with (directory / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    write_json(directory / "result.json", report)
    return report
