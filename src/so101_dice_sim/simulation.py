from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np

from .faces import identify_top_face
from .settings import Settings


@dataclass(frozen=True)
class Sample:
    time_s: float
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity_m_s: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]
    table_contact: bool
    bottom_gap_m: float
    within_table: bool
    top_face: int | None
    tilt_deg: float


@dataclass(frozen=True)
class SimulationResult:
    samples: list[Sample]
    warnings: dict[str, int]
    model: dict


def run_simulation(settings: Settings) -> SimulationResult:
    model = mujoco.MjModel.from_xml_path(settings.scene)
    model.opt.timestep = settings.timestep_s
    dice = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    dice_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dice_geom")
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free")
    if min(dice, dice_geom, table, joint) < 0:
        raise ValueError("MJCFにdice、dice_geom、table、dice_freeが必要です")
    if (
        model.nq != 7 or model.nv != 6
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_bodyid[joint] != dice
        or model.geom_bodyid[dice_geom] != dice
        or model.geom_bodyid[table] != 0
        or any(
            model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX
            for g in (dice_geom, table)
        )
        or not np.allclose(model.geom_pos[dice_geom], 0)
        or not np.allclose(model.geom_quat[[dice_geom, table]], [1, 0, 0, 0])
        or not np.allclose(model.body_ipos[dice], 0)
    ):
        raise ValueError("この最小版は中心一致の自由立方体と軸整列した固定天板に対応します")
    half_size = model.geom_size[dice_geom].copy()
    if (
        not np.isfinite(half_size).all() or np.any(half_size <= 0)
        or not np.allclose(half_size, half_size[0])
        or not math.isfinite(float(model.body_mass[dice]))
        or model.body_mass[dice] <= 0
    ):
        raise ValueError("サイコロの寸法と質量は有限の正数で、形状は立方体が必要です")
    data = mujoco.MjData(model)
    address = model.jnt_qposadr[joint]
    data.qpos[address:address + 3] = settings.initial_position_m
    data.qpos[address + 3:address + 7] = settings.initial_quaternion_wxyz
    mujoco.mj_forward(model, data)
    table_center = data.geom_xpos[table].copy()
    table_half = model.geom_size[table].copy()
    table_top = float(table_center[2] + table_half[2])

    def observe() -> Sample:
        if not all(
            np.isfinite(value).all()
            for value in (data.qpos, data.qvel, data.qacc)
        ):
            raise FloatingPointError("MuJoCoの状態に非有限値が発生しました")
        rotation = data.xmat[dice].reshape(3, 3)
        position = data.xpos[dice]
        extent = np.abs(rotation) @ half_size
        velocity = np.empty(6)
        # mj_objectVelocityは回転、並進の順。flg_local=0でワールド座標。
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, dice, velocity, 0
        )
        face, tilt = identify_top_face(rotation, settings.face_tolerance_deg)
        contact = any(
            {int(c.geom1), int(c.geom2)} == {dice_geom, table}
            and c.dist <= settings.height_tolerance_m
            for c in data.contact
        )
        return Sample(
            time_s=float(data.time),
            position_m=tuple(float(v) for v in position),
            quaternion_wxyz=tuple(float(v) for v in data.xquat[dice]),
            linear_velocity_m_s=tuple(float(v) for v in velocity[3:]),
            angular_velocity_rad_s=tuple(float(v) for v in velocity[:3]),
            table_contact=contact,
            bottom_gap_m=float(position[2] - extent[2] - table_top),
            within_table=bool(np.all(
                np.abs(position[:2] - table_center[:2]) + extent[:2]
                <= table_half[:2]
            )),
            top_face=face,
            tilt_deg=tilt,
        )

    samples = [observe()]
    if samples[0].bottom_gap_m < 0:
        raise ValueError("初期状態でサイコロが天板の高さより下にあります")
    for _ in range(settings.steps):
        mujoco.mj_step(model, data)
        # mj_step後の積分済みqpos/qvelに観測・接触状態を同期する。
        mujoco.mj_forward(model, data)
        samples.append(observe())
    warnings = {
        mujoco.mjtWarning(index).name: int(warning.number)
        for index, warning in enumerate(data.warning)
        if warning.number
    }
    return SimulationResult(samples, warnings, {
        "dice_edge_m": float(2 * half_size[0]),
        "dice_mass_kg": float(model.body_mass[dice]),
        "table_top_m": table_top,
        "gravity_m_s2": model.opt.gravity.tolist(),
        "timestep_s": float(model.opt.timestep),
    })


def validate_result(result: SimulationResult, settings: Settings) -> dict:
    samples = result.samples
    window = samples[-(settings.window_steps + 1):]
    finite = all(
        math.isfinite(value)
        for sample in samples
        for value in (
            sample.time_s, *sample.position_m, *sample.quaternion_wxyz,
            *sample.linear_velocity_m_s, *sample.angular_velocity_rad_s,
            sample.bottom_gap_m, sample.tilt_deg,
        )
    )
    max_linear = max(
        (math.hypot(*s.linear_velocity_m_s) for s in window), default=None
    )
    max_angular = max(
        (math.hypot(*s.angular_velocity_rad_s) for s in window), default=None
    )
    checks = {
        "finite_values": finite,
        "sample_count": len(samples) == settings.steps + 1,
        "simulation_times": bool(samples) and all(
            math.isclose(s.time_s, i * settings.timestep_s, abs_tol=1e-9, rel_tol=0)
            for i, s in enumerate(samples)
        ),
        "no_mujoco_warnings": not result.warnings,
        "linear_speed": finite and max_linear is not None
        and max_linear < settings.linear_speed_limit_m_s,
        "angular_speed": finite and max_angular is not None
        and max_angular < settings.angular_speed_limit_rad_s,
        "table_contact": bool(window) and all(s.table_contact for s in window),
        "within_table": bool(window) and all(s.within_table for s in window),
        "bottom_height": bool(window) and all(
            abs(s.bottom_gap_m) <= settings.height_tolerance_m for s in window
        ),
        "top_face_determined": bool(window) and all(
            s.top_face in (1, 2, 3, 4, 5, 6)
            and 0 <= s.tilt_deg <= settings.face_tolerance_deg + 1e-10
            for s in window
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "window_start_s": settings.duration_s - settings.settle_window_s,
        "window_end_s": settings.duration_s,
        "window_samples": len(window),
        "max_linear_speed_m_s": max_linear,
        "max_angular_speed_rad_s": max_angular,
        "max_abs_bottom_gap_m": max(
            (abs(s.bottom_gap_m) for s in window), default=None
        ),
        "final_state": asdict(samples[-1]) if samples else None,
    }
