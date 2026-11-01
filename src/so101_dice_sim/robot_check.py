import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .reporting import write_json


JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
SOURCE_COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
TRACKING_LIMIT_RAD = 0.01
SPEED_LIMIT_RAD_S = 0.01
ROOT = Path(__file__).resolve().parents[2]


def verify_assets(directory: Path) -> dict:
    with (directory / "provenance.json").open(encoding="utf-8-sig") as stream:
        provenance = json.load(stream)
    if provenance["commit"] != SOURCE_COMMIT:
        raise ValueError("SO-101の取得コミットが想定と一致しません")
    for entry in provenance["files"]:
        actual = hashlib.sha256((directory / entry["path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"SO-101資産のハッシュ不一致: {entry['path']}")
    return provenance


def validate_robot_samples(
    samples: list[dict], ranges: np.ndarray, force_ranges: np.ndarray,
    warning_count: int,
) -> dict:
    if not samples:
        raise ValueError("SO-101検証サンプルがありません")
    positions = np.array([s["qpos"] for s in samples])
    velocities = np.array([s["qvel"] for s in samples])
    forces = np.array([s["force"] for s in samples])
    targets = np.array([s["target"] for s in samples])
    settled = np.array([s["settled"] for s in samples])
    finite = bool(all(np.isfinite(a).all() for a in (
        positions, velocities, forces, targets,
        np.array([s["tip_surface_distance_m"] for s in samples]),
    )))
    tracking = float(np.max(np.abs(positions[settled] - targets[settled])))
    speed = float(np.max(np.abs(velocities[settled])))
    displacement = np.ptp(positions, axis=0)
    checks = {
        "finite_values": finite,
        "no_mujoco_warnings": warning_count == 0,
        "no_contacts": all(s["contact_count"] == 0 for s in samples),
        "joint_limits": bool(np.all(positions >= ranges[:, 0] - 1e-6)
                             and np.all(positions <= ranges[:, 1] + 1e-6)),
        "force_limits": bool(np.all(forces >= force_ranges[:, 0] - 1e-6)
                             and np.all(forces <= force_ranges[:, 1] + 1e-6)),
        "tracking": finite and tracking < TRACKING_LIMIT_RAD,
        "settled_speed": finite and speed < SPEED_LIMIT_RAD_S,
        "all_arm_joints_moved": bool(np.all(displacement[:5] > 0.05)),
        "gripper_opened_and_closed": bool(
            displacement[5] > 0.9 and abs(positions[-1, 5]) < TRACKING_LIMIT_RAD
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_reasons": [key for key, value in checks.items() if not value],
        "tracking_limit_rad": TRACKING_LIMIT_RAD,
        "speed_limit_rad_s": SPEED_LIMIT_RAD_S,
        "max_settled_tracking_error_rad": tracking,
        "max_settled_speed_rad_s": speed,
        "joint_excursion_rad": displacement.tolist(),
        "max_contact_count": max(s["contact_count"] for s in samples),
    }


def run_robot_check(directory: Path) -> dict:
    assets = ROOT / "assets"
    provenance = verify_assets(assets / "robotstudio_so101")
    model = mujoco.MjModel.from_xml_path(str(assets / "so101_scene.xml"))
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("SO-101は6関節・6アクチュエーターを想定しています")
    for index, name in enumerate(JOINT_NAMES):
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) != name
            or model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE
            or model.actuator_trnid[index, 0] != index
            or not model.jnt_limited[index]
            or not model.actuator_ctrllimited[index]
            or not model.actuator_forcelimited[index]
        ):
            raise ValueError(f"想定と異なる関節・アクチュエーター構成: {name}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_contacts = [
        {
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            or f"geom-{c.geom1}",
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            or f"geom-{c.geom2}",
            "distance_m": float(c.dist),
        }
        for c in data.contact
    ]
    tips = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")
    ]
    if min(tips) < 0:
        raise ValueError("指先距離の観測対象geomが見つかりません")
    arm_target = np.array([0.1, 0.1, -0.1, 0.1, 0.1, 0.0])
    open_target = arm_target.copy()
    open_target[5] = 1.0
    phases = [
        ("hold", np.zeros(6), 2.0),
        ("arm_move", arm_target, 3.0),
        ("open", open_target, 3.0),
        ("close", arm_target, 3.0),
    ]
    for _, target, _ in phases:
        if np.any(target < model.actuator_ctrlrange[:, 0]) or np.any(
            target > model.actuator_ctrlrange[:, 1]
        ):
            raise ValueError("検証目標がアクチュエーターの範囲外です")
    samples = []
    phase_results = []
    previous = np.zeros(6)
    for phase, target, duration in phases:
        phase_start = float(data.time)
        for step in range(round(duration / model.opt.timestep)):
            elapsed = (step + 1) * model.opt.timestep
            fraction = min(elapsed / 1.0, 1.0)
            smooth = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[:] = previous + smooth * (target - previous)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not all(np.isfinite(a).all() for a in (
                data.qpos, data.qvel, data.qacc, data.actuator_force,
            )):
                raise FloatingPointError("SO-101の物理状態に非有限値が発生しました")
            tip_distance = float(
                np.linalg.norm(data.geom_xpos[tips[0]] - data.geom_xpos[tips[1]])
                - sum(model.geom_size[g, 0] for g in tips)
            )
            samples.append({
                "time_s": float(data.time),
                "phase": phase,
                "settled": elapsed >= duration - 0.5,
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(),
                "target": data.ctrl.tolist(),
                "force": data.actuator_force.tolist(),
                "contact_count": int(data.ncon),
                "tip_surface_distance_m": tip_distance,
            })
        phase_results.append({
            "phase": phase, "start_s": phase_start, "end_s": float(data.time),
            "target_rad": target.tolist(), "actual_rad": data.qpos.tolist(),
            "tip_surface_distance_m": samples[-1]["tip_surface_distance_m"],
        })
        previous = target
    warnings = {
        mujoco.mjtWarning(index).name: int(warning.number)
        for index, warning in enumerate(data.warning) if warning.number
    }
    validation = validate_robot_samples(
        samples, model.jnt_range, model.actuator_forcerange, sum(warnings.values())
    )
    validation["checks"]["initial_no_contacts"] = not initial_contacts
    validation["passed"] = all(validation["checks"].values())
    validation["failure_reasons"] = [
        key for key, value in validation["checks"].items() if not value
    ]
    report = {
        "task": "so101_standalone_check", "schema_version": 1,
        "mujoco_version": mujoco.__version__,
        "source": provenance,
        "units": {"angle": "rad", "speed": "rad/s", "distance": "m", "force": "N*m"},
        "duration_s": float(data.time), "timestep_s": float(model.opt.timestep),
        "sample_count": len(samples),
        "joints": [
            {
                "name": name, "range_rad": model.jnt_range[i].tolist(),
                "ctrlrange_rad": model.actuator_ctrlrange[i].tolist(),
                "forcerange_nm": model.actuator_forcerange[i].tolist(),
                "axis_local": model.jnt_axis[i].tolist(),
            } for i, name in enumerate(JOINT_NAMES)
        ],
        "initial_contacts": initial_contacts, "warnings": warnings,
        "phases": phase_results, "validation": validation,
        "limitations": [
            "アーム単体と床の検証。テーブル・サイコロとの干渉、把持は未検証。",
            "指先距離は特定の2球の表面間の直線距離で、有効開口幅ではない。",
            "固定基部の衝突形状は上流モデルに含まれず、基部と床の接触は評価しない。",
            "全可動域・全姿勢の衝突検査ではなく、小さな固定軌道の検証。",
        ],
    }
    with (directory / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for sample in samples:
            row = {key: value for key, value in sample.items()
                   if key not in ("qpos", "qvel", "target", "force")}
            for kind in ("qpos", "qvel", "target", "force"):
                row.update({
                    f"{name}_{kind}": value
                    for name, value in zip(JOINT_NAMES, sample[kind])
                })
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
    write_json(directory / "result.json", report)
    return report
