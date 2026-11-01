import math

import numpy as np

from .approach import TIMESTEP_S
from .placement import placement_timing


ROTATION_START_S = 16.0
ROTATION_HOLD_START_S = 20.0
ROTATION_END_S = 22.0
TARGET_FACE = 2


def validate_rotation(
    samples: list[dict], warnings: dict, *, target_face: int = TARGET_FACE,
    observe_position: bool = False,
) -> dict:
    timing = placement_timing(target_face, observe_position=observe_position)
    delta = 0.0 if target_face == 1 else -math.pi / 2
    roll_delta = -math.pi / 2 if target_face == 3 else 0.0
    if not samples or not any(s["phase"] == "rotation_hold" for s in samples):
        raise ValueError("回転後の保持サンプルがありません")
    rotation = [s for s in samples if s["time_s"] >= timing.grasp_end_s - 1e-9]
    hold = [s for s in samples if s["phase"] == "rotation_hold"]
    relative = np.array([
        [s[f"dice_in_gripper_{axis}_m"] for axis in "xyz"] for s in rotation
    ])
    drift = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
    checks = {
        "finite_values": all(
            math.isfinite(value) for s in samples for value in s.values()
            if isinstance(value, (int, float))
        ),
        "sample_count": len(samples) == round(timing.rotation_end_s / TIMESTEP_S) + 1,
        "simulation_times": all(
            math.isclose(s["time_s"], i * TIMESTEP_S, abs_tol=1e-9, rel_tol=0)
            for i, s in enumerate(samples)
        ),
        "full_rotation_window": len(rotation) == round(
            (timing.rotation_end_s - timing.grasp_end_s) / TIMESTEP_S
        ) + 1,
        "full_hold_window": len(hold) == 1001 and all(
            (s["phase"] == "rotation_hold") == (
                i * TIMESTEP_S >= timing.rotation_hold_start_s
            )
            for i, s in enumerate(samples)
        ),
        "no_mujoco_warnings": not warnings,
        "no_unexpected_contacts": all(s["unexpected_contacts"] == 0 for s in samples),
        "joint_and_force_limits": all(s["limits_ok"] for s in samples),
        "joint_margin": all(s["min_arm_joint_margin_rad"] >= 0.1 for s in samples),
        "both_jaws_through_rotation": all(
            s["fixed_jaw_force_n"] >= 0.01 and s["moving_jaw_force_n"] >= 0.01
            for s in rotation
        ),
        "off_table_through_rotation": all(
            not s["dice_table_contact"] and s["dice_bottom_gap_m"] >= 0.03
            for s in rotation
        ),
        "no_position_slip": math.isfinite(drift) and drift < 0.002,
        "requested_face_up": all(
            s["top_face"] == target_face and 0 <= s["tilt_deg"] <= 5 for s in hold
        ),
        "rotation_actually_executed": all(
            abs(
                (s["wrist_flex_qpos"] - rotation[0]["wrist_flex_qpos"])
                - delta
            ) < 0.01 and (
                target_face != 3 or abs(
                    s["wrist_roll_qpos"] - rotation[0]["wrist_roll_qpos"] - roll_delta
                ) < 0.01
            ) for s in hold
        ),
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
        "requested_face": target_face,
        "required_wrist_flex_delta_rad": delta,
        "required_wrist_roll_delta_rad": roll_delta,
        "hold_start_s": timing.rotation_hold_start_s,
        "hold_duration_s": timing.rotation_end_s - timing.rotation_hold_start_s,
        "hold_samples": len(hold),
        "rotation_samples": len(rotation),
        "max_tilt_deg": max(s["tilt_deg"] for s in hold),
        "max_relative_drift_m": drift,
        "relative_drift_limit_m": 0.002,
        "min_clearance_m": min(s["dice_bottom_gap_m"] for s in rotation),
        "required_clearance_m": 0.03,
        "min_fixed_jaw_force_n": min(s["fixed_jaw_force_n"] for s in rotation),
        "min_moving_jaw_force_n": min(s["moving_jaw_force_n"] for s in rotation),
        "required_jaw_force_n": 0.01,
    }
