import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from .endpoint_collection import load_endpoint_replay
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES, ROOT, verify_assets


TIMESTEP_S = 0.002
DISTANCE_CAP_M = 2.0
SCENE_PATH = ROOT / "assets" / "so101_dice_scene.xml"
POLICY_COLUMNS = {
    "time_s", "phase", "unexpected_contacts",
    *(f"{name}_{suffix}" for name in JOINT_NAMES
      for suffix in ("qpos", "qvel", "target")),
    *(f"dice_{axis}_m" for axis in "xyz"),
    *(f"dice_quat_{axis}" for axis in "wxyz"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time_axis(times: np.ndarray, timestep_s: float, duration_s: float) -> None:
    values = np.asarray(times, dtype=np.float64)
    if (
        values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
        or not np.isfinite(timestep_s) or timestep_s <= 0
        or not np.isfinite(duration_s) or duration_s <= 0
    ):
        raise ValueError("trajectoryの時刻または刻みが不正です")
    expected = np.arange(len(values), dtype=np.float64) * timestep_s
    if (
        not np.allclose(values, expected, atol=1e-9, rtol=0)
        or not np.isclose(values[-1], duration_s, atol=1e-9, rtol=0)
    ):
        raise ValueError("trajectoryの時刻が共通のMuJoCo時間軸と一致しません")


def _vector(row: dict, fields: list[str], label: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[field]) for field in fields], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}の列または値が不正です") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label}に非有限値があります")
    return values


def _policy_pose(row: dict) -> np.ndarray:
    arm = _vector(row, [f"{name}_qpos" for name in JOINT_NAMES], "関節姿勢")
    position = _vector(row, [f"dice_{axis}_m" for axis in "xyz"], "サイコロ位置")
    quaternion = _vector(
        row, [f"dice_quat_{axis}" for axis in "wxyz"], "サイコロ姿勢",
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-8, rtol=0):
        raise ValueError("サイコロのクォータニオンが単位長ではありません")
    return np.concatenate((arm, position, quaternion))


def _joint_vector(row: dict, suffix: str) -> np.ndarray:
    return _vector(
        row, [f"{name}_{suffix}" for name in JOINT_NAMES], f"関節{suffix}",
    )


def _read_policy_trajectory(path: Path, evaluation: dict) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not POLICY_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("SmolVLA trajectoryに必要な列がありません")
        rows = list(reader)
    steps = evaluation.get("physics_steps")
    duration = evaluation.get("duration_s")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("評価JSONのphysics_stepsが不正です")
    if len(rows) != steps + 1:
        raise ValueError("評価JSONとtrajectory.csvのサンプル数が一致しません")
    times = _vector_rows(rows, "time_s")
    _time_axis(times, TIMESTEP_S, float(duration))
    for index, row in enumerate(rows):
        _policy_pose(row)
        _joint_vector(row, "qvel")
        _joint_vector(row, "target")
        count = row.get("unexpected_contacts", "")
        try:
            parsed_count = int(count)
        except (TypeError, ValueError) as error:
            raise ValueError("unexpected_contactsの値が不正です") from error
        if parsed_count < 0:
            raise ValueError("unexpected_contactsは負にできません")
        if not row.get("phase"):
            raise ValueError("trajectoryのphaseが空です")
        if index == 0 and not np.array_equal(
            _policy_pose(row), np.asarray(evaluation["initial_qpos"], dtype=np.float64),
        ):
            raise ValueError("trajectoryの初期状態と評価JSONが一致しません")
    return rows


def _vector_rows(rows: list[dict], field: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"trajectoryの{field}列が不正です") from error
    if not np.isfinite(values).all():
        raise ValueError(f"trajectoryの{field}列に非有限値があります")
    return values


def _reported_contact_pairs(failure_reasons: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for reason in failure_reasons:
        prefix = "unexpected_contacts:"
        if reason.startswith(prefix):
            try:
                raw_pairs = ast.literal_eval(reason[len(prefix):])
            except (SyntaxError, ValueError) as error:
                raise ValueError("評価JSONの想定外接触ペアが不正です") from error
            if not isinstance(raw_pairs, list):
                raise ValueError("評価JSONの想定外接触ペアが配列ではありません")
            for pair in raw_pairs:
                if (
                    not isinstance(pair, tuple) or len(pair) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in pair)
                ):
                    raise ValueError("評価JSONの想定外接触geom IDが不正です")
                pairs.append(tuple(sorted(pair)))
    if not pairs:
        raise ValueError("評価JSONに想定外接触の停止理由がありません")
    return sorted(set(pairs))


def _measure_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    table_geom: int,
    jaw_geom: int,
) -> tuple[float, bool]:
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (model.nq,) or not np.isfinite(values).all():
        raise ValueError("静的幾何評価のqpos形状または値が不正です")
    data.qpos[:] = values
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    distance = float(mujoco.mj_geomDistance(
        model, data, table_geom, jaw_geom, DISTANCE_CAP_M, None,
    ))
    if not np.isfinite(distance) or distance >= DISTANCE_CAP_M:
        raise ValueError("geom間距離が上限内で算出できません")
    pair = tuple(sorted((table_geom, jaw_geom)))
    in_contact = any(
        tuple(sorted((data.contact[index].geom1, data.contact[index].geom2))) == pair
        for index in range(data.ncon)
    )
    return distance, in_contact


def _clearance_summary(times: np.ndarray, distances: np.ndarray) -> dict:
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != times.shape or not np.isfinite(values).all():
        raise ValueError("geom距離の時系列が不正です")
    minimum_index = int(np.argmin(values))
    colliding = np.flatnonzero(values <= 0)
    return {
        "minimum_distance_m": float(values[minimum_index]),
        "minimum_distance_time_s": float(times[minimum_index]),
        "first_nonpositive_distance_time_s": (
            float(times[int(colliding[0])]) if len(colliding) else None
        ),
        "initial_distance_m": float(values[0]),
        "final_distance_m": float(values[-1]),
    }


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is None:
        raise ValueError(f"MuJoCo geom名がありません: {geom_id}")
    return name


def _input_fingerprints(
    evaluation_path: Path, teacher_path: Path, provenance: dict,
) -> dict:
    files = {
        "evaluation.json": evaluation_path / "evaluation.json",
        "settings.json": evaluation_path.parent / "settings.json",
        "policy_trajectory.csv": evaluation_path / "trajectory.csv",
        "teacher_replay.json": teacher_path / "replay.json",
        "teacher_trajectory.csv": teacher_path / "trajectory.csv",
        "scene_mjcf": SCENE_PATH,
        "robot_asset_provenance": ROOT / "assets" / "robotstudio_so101" / "provenance.json",
    }
    for item in provenance["files"]:
        files[f"robot_asset:{item['path']}"] = ROOT / "assets" / "robotstudio_so101" / item["path"]
    return {
        "files_sha256": {name: _sha256(path) for name, path in sorted(files.items())},
        "teacher_path": str(teacher_path.resolve()),
        "evaluation_path": str(evaluation_path.resolve()),
    }


def analyze_saved_contact(directory: Path, evaluation_path: Path) -> dict:
    evaluation_path = evaluation_path.resolve()
    evaluation_file = evaluation_path / "evaluation.json"
    settings_file = evaluation_path.parent / "settings.json"
    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    if (
        evaluation.get("evaluation_type") != "training_position_diagnostic"
        or evaluation.get("requested_face") != 2
        or evaluation.get("offset_mm") != [0.0, 0.0]
        or evaluation.get("passed") is not False
        or evaluation.get("completed") is not False
        or settings.get("evaluation_type") != "training_position_diagnostic"
        or settings.get("target_face") != 2
        or settings.get("evaluation_offset_mm") != [0.0, 0.0]
    ):
        raise ValueError("保存済み中央位置・面2の失敗評価ではありません")
    failure_reasons = evaluation.get("failure_reasons")
    if not isinstance(failure_reasons, list) or not all(
        isinstance(reason, str) for reason in failure_reasons
    ):
        raise ValueError("評価JSONのfailure_reasonsが不正です")
    contact_pairs = _reported_contact_pairs(failure_reasons)
    teacher_path = Path(settings["teacher"])
    if teacher_path.name != "train-04-endpoint":
        raise ValueError("中央学習位置のtrain-04成功教師ではありません")
    rows = _read_policy_trajectory(evaluation_path / "trajectory.csv", evaluation)
    teacher, _ = load_endpoint_replay(teacher_path)
    initial_qpos = np.asarray(evaluation.get("initial_qpos"), dtype=np.float64)
    if (
        initial_qpos.shape != (13,) or not np.isfinite(initial_qpos).all()
        or not np.array_equal(initial_qpos, teacher.qpos[0])
        or not np.array_equal(initial_qpos, np.asarray(settings["initial_qpos"], dtype=np.float64))
    ):
        raise ValueError("SmolVLA評価と成功教師の初期qposが一致しません")
    times = _vector_rows(rows, "time_s")
    teacher_indices = np.rint(times / TIMESTEP_S).astype(np.int64)
    if (
        np.any(teacher_indices >= len(teacher.qpos))
        or not np.allclose(teacher.times[teacher_indices], times, atol=1e-9, rtol=0)
    ):
        raise ValueError("教師軌道を失敗評価と同じ時刻へ整列できません")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    if (model.nq, model.nv, model.nu) != (13, 12, 6):
        raise ValueError("診断用MuJoCoモデルのqpos/qvel/actuator構成が想定外です")
    table_geom = model.geom("table").id
    jaw_geom = model.geom("fixed_jaw_box3").id
    actual_pair = tuple(sorted((table_geom, jaw_geom)))
    if actual_pair not in contact_pairs:
        raise ValueError("評価停止ペアがtableとfixed_jaw_box3ではありません")
    pair_names = [_geom_name(model, geom_id) for geom_id in actual_pair]
    pair_bodies = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        for geom_id in actual_pair
    ]

    policy_distances = []
    teacher_distances = []
    policy_pair_contacts = []
    teacher_pair_contacts = []
    comparison = []
    data = mujoco.MjData(model)
    for index, (row, teacher_index) in enumerate(zip(rows, teacher_indices)):
        policy_qpos = _policy_pose(row)
        teacher_qpos = teacher.qpos[teacher_index]
        policy_distance, policy_contact = _measure_clearance(
            model, data, policy_qpos, table_geom, jaw_geom,
        )
        teacher_distance, teacher_contact = _measure_clearance(
            model, data, teacher_qpos, table_geom, jaw_geom,
        )
        policy_q = policy_qpos[:6]
        teacher_q = teacher_qpos[:6]
        policy_target = _joint_vector(row, "target")
        teacher_target = _joint_vector(teacher.rows[teacher_index], "target")
        policy_velocity = _joint_vector(row, "qvel")
        teacher_velocity = _joint_vector(teacher.rows[teacher_index], "qvel")
        policy_xyz = policy_qpos[6:9]
        teacher_xyz = teacher_qpos[6:9]
        policy_distances.append(policy_distance)
        teacher_distances.append(teacher_distance)
        policy_pair_contacts.append(policy_contact)
        teacher_pair_contacts.append(teacher_contact)
        comparison.append({
            "time_s": f"{times[index]:.6f}",
            "policy_phase": row["phase"],
            "teacher_phase": teacher.rows[teacher_index]["phase"],
            "policy_unexpected_contacts": int(row["unexpected_contacts"]),
            "policy_fixed_jaw_table_clearance_m": f"{policy_distance:.12g}",
            "teacher_fixed_jaw_table_clearance_m": f"{teacher_distance:.12g}",
            "policy_pair_in_contact_after_mj_forward": policy_contact,
            "teacher_pair_in_contact_after_mj_forward": teacher_contact,
            "policy_joint_qpos_rad": json.dumps(policy_q.tolist()),
            "teacher_joint_qpos_rad": json.dumps(teacher_q.tolist()),
            "joint_state_error_rad": json.dumps((policy_q - teacher_q).tolist()),
            "policy_joint_qvel_rad_s": json.dumps(policy_velocity.tolist()),
            "teacher_joint_qvel_rad_s": json.dumps(teacher_velocity.tolist()),
            "policy_joint_target_rad": json.dumps(policy_target.tolist()),
            "teacher_joint_target_rad": json.dumps(teacher_target.tolist()),
            "joint_target_error_rad": json.dumps((policy_target - teacher_target).tolist()),
            "policy_joint_tracking_error_rad": json.dumps((policy_q - policy_target).tolist()),
            "teacher_joint_tracking_error_rad": json.dumps((teacher_q - teacher_target).tolist()),
            "policy_dice_xyz_m": json.dumps(policy_xyz.tolist()),
            "teacher_dice_xyz_m": json.dumps(teacher_xyz.tolist()),
            "dice_position_error_m": json.dumps((policy_xyz - teacher_xyz).tolist()),
            "policy_gripper_qpos_rad": f"{policy_q[5]:.12g}",
            "teacher_gripper_qpos_rad": f"{teacher_q[5]:.12g}",
            "policy_gripper_target_rad": f"{policy_target[5]:.12g}",
            "teacher_gripper_target_rad": f"{teacher_target[5]:.12g}",
        })

    first_recorded_contact = next(
        (index for index, row in enumerate(rows) if int(row["unexpected_contacts"]) > 0),
        None,
    )
    if first_recorded_contact is None:
        raise ValueError("評価JSONに想定外接触がありますがtrajectoryに記録されていません")
    contact_index = first_recorded_contact
    if not policy_pair_contacts[contact_index]:
        raise ValueError("保存された接触時刻のmj_forwardで接触ペアを確認できません")
    policy_distances = np.asarray(policy_distances, dtype=np.float64)
    teacher_distances = np.asarray(teacher_distances, dtype=np.float64)
    first_forward_contact = np.flatnonzero(policy_distances <= 0)
    contact_row = rows[contact_index]
    policy_q = _policy_pose(contact_row)[:6]
    teacher_q = teacher.qpos[teacher_indices[contact_index], :6]
    policy_target = _joint_vector(contact_row, "target")
    teacher_target = _joint_vector(teacher.rows[teacher_indices[contact_index]], "target")

    csv_path = directory / "trajectory_comparison.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    provenance = verify_assets(ROOT / "assets" / "robotstudio_so101")
    report = {
        "status": "complete",
        "diagnostic_only": True,
        "training_performed": False,
        "physics_replay_executed": False,
        "physics_steps_executed": 0,
        "mj_step_called": False,
        "mj_forward_calls": 2 * len(rows),
        "uploaded": False,
        "mujoco_version": mujoco.__version__,
        "evaluation": {
            "path": str(evaluation_path),
            "duration_s": float(times[-1]),
            "saved_physics_steps": int(evaluation["physics_steps"]),
            "failure_reasons": failure_reasons,
            "requested_face": 2,
            "initial_top_face": 1,
        },
        "teacher": {
            "path": str(teacher_path.resolve()),
            "mode": teacher.report["mode"],
            "passed": teacher.report["passed"],
            "requested_face": teacher.report["requested_face"],
            "initial_top_face": int(teacher.rows[0]["top_face"]),
        },
        "model": {
            "scene": str(SCENE_PATH),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "table_geom": {"id": table_geom, "name": _geom_name(model, table_geom)},
            "fixed_jaw_geom": {
                "id": jaw_geom, "name": _geom_name(model, jaw_geom),
            },
            "reported_contact_pair": {
                "geom_ids": list(actual_pair),
                "geom_names": pair_names,
                "body_names": pair_bodies,
            },
            "asset_source_commit": provenance["commit"],
        },
        "comparison": {
            "sample_count": len(rows),
            "timestep_s": TIMESTEP_S,
            "time_alignment": "same elapsed MuJoCo time; both start at the identical initial qpos",
            "first_recorded_unexpected_contact": {
                "time_s": float(times[contact_index]),
                "policy_phase": contact_row["phase"],
                "teacher_phase": teacher.rows[teacher_indices[contact_index]]["phase"],
                "unexpected_contact_count": int(contact_row["unexpected_contacts"]),
                "fixed_jaw_table_clearance_m": float(policy_distances[contact_index]),
                "pair_present_after_mj_forward": policy_pair_contacts[contact_index],
            },
            "first_nonpositive_mj_forward_distance_time_s": (
                float(times[int(first_forward_contact[0])]) if len(first_forward_contact) else None
            ),
            "policy_fixed_jaw_table_clearance": _clearance_summary(times, policy_distances),
            "teacher_fixed_jaw_table_clearance": _clearance_summary(times, teacher_distances),
            "at_first_recorded_contact": {
                "policy_joint_qpos_rad": policy_q.tolist(),
                "teacher_joint_qpos_rad": teacher_q.tolist(),
                "joint_state_error_rad": (policy_q - teacher_q).tolist(),
                "policy_joint_target_rad": policy_target.tolist(),
                "teacher_joint_target_rad": teacher_target.tolist(),
                "joint_target_error_rad": (policy_target - teacher_target).tolist(),
                "policy_joint_tracking_error_rad": (policy_q - policy_target).tolist(),
                "teacher_joint_tracking_error_rad": (teacher_q - teacher_target).tolist(),
                "policy_gripper_qpos_rad": float(policy_q[5]),
                "teacher_gripper_qpos_rad": float(teacher_q[5]),
                "policy_gripper_target_rad": float(policy_target[5]),
                "teacher_gripper_target_rad": float(teacher_target[5]),
            },
        },
        "inputs": _input_fingerprints(evaluation_path, teacher_path, provenance),
        "interpretation_limits": [
            "geom距離は保存姿勢からmj_forward後に計算した静的値で、接触solverを物理再生した結果ではありません。",
            "同時刻の教師との差は観測事実であり、接触の因果関係を単独で証明しません。",
            "mj_forwardの接触判定は保存済みunexpected_contacts数とは別に記録した診断値です。",
        ],
    }
    write_json(directory / "contact_diagnostics.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="保存済みSmolVLA失敗軌道の接触を非物理・オフライン診断",
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/learning/contact-diagnostics"),
    )
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        report = analyze_saved_contact(directory, args.evaluation)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        write_json(directory / "error.json", {
            "status": "error", "error_type": type(error).__name__, "message": str(error),
        })
        print(f"接触オフライン診断エラー: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    comparison = report["comparison"]
    contact = comparison["first_recorded_unexpected_contact"]
    print(f"接触オフライン診断保存先: {directory.resolve()}", flush=True)
    print(
        f"最初の記録接触: {contact['time_s']:.3f}s, "
        f"固定指-テーブル距離 {contact['fixed_jaw_table_clearance_m']:.6g}m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
