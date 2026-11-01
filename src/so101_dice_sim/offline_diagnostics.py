import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from .action_contract import ENDPOINT_MODE, dataset_fingerprint
from .endpoint_collection import load_endpoint_replay
from .learning import (
    FPS, bounded_action, checkpoint_fingerprint, load_policy, training_normalization_stats,
)
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES, ROOT
from .training_position_evaluation import load_training_position


CHUNK_SIZE = 10
PHYSICS_TIMESTEP_S = .002
DIAGNOSTIC_SEED = 42
DATASET_TIMESTAMP_TOLERANCE_S = 1e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_fingerprint(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_errors(errors: np.ndarray) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(JOINT_NAMES):
        raise ValueError("action誤差はサンプル×6関節の配列である必要があります")
    if not np.isfinite(values).all():
        raise ValueError("action誤差に非有限値があります")
    absolute = np.abs(values)
    return {
        "sample_count": int(values.shape[0]),
        "scalar_count": int(values.size),
        "mae_rad": float(absolute.mean()) if values.size else 0.0,
        "rmse_rad": float(np.sqrt(np.square(values).mean())) if values.size else 0.0,
        "max_abs_rad": float(absolute.max()) if values.size else 0.0,
        "per_joint": {
            name: {
                "mae_rad": float(absolute[:, index].mean()) if len(values) else 0.0,
                "rmse_rad": float(np.sqrt(np.square(values[:, index]).mean())) if len(values) else 0.0,
                "max_abs_rad": float(absolute[:, index].max()) if len(values) else 0.0,
            }
            for index, name in enumerate(JOINT_NAMES)
        },
    }


def _quat_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("クォータニオンは4要素である必要があります")
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if not np.isfinite(left).all() or not np.isfinite(right).all() or min(left_norm, right_norm) <= 0:
        raise ValueError("クォータニオンが不正です")
    dot = abs(float(np.dot(left / left_norm, right / right_norm)))
    return float(2 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _target(row: dict, prefix: str = "") -> np.ndarray:
    return np.array([float(row[f"{prefix}{name}_target"]) for name in JOINT_NAMES], dtype=np.float64)


def _aligned_sample_time(timestamp: float, frame_index: int) -> float:
    expected = frame_index / FPS
    if not np.isfinite(timestamp) or not np.isclose(
        timestamp, expected, atol=DATASET_TIMESTAMP_TOLERANCE_S, rtol=0,
    ):
        raise ValueError("dataset timestampと教師500Hz indexが一致しません")
    return expected


def _validate_action_chunk(
    target_chunk: np.ndarray,
    pad_chunk: np.ndarray,
    episode_actions: np.ndarray,
    frame_index: int,
) -> tuple[int, int]:
    target_chunk = np.asarray(target_chunk)
    pad_chunk = np.asarray(pad_chunk, dtype=bool)
    episode_actions = np.asarray(episode_actions)
    if (
        target_chunk.ndim != 2 or target_chunk.shape[1] != len(JOINT_NAMES)
        or pad_chunk.shape != (target_chunk.shape[0],)
        or episode_actions.ndim != 2 or episode_actions.shape[1] != len(JOINT_NAMES)
        or frame_index < 0
    ):
        raise ValueError("教師action chunkまたはepisode境界の形状が不正です")
    valid_count = 0
    padded_count = 0
    for step, is_pad in enumerate(pad_chunk):
        source_index = frame_index + step
        if is_pad:
            if source_index < len(episode_actions):
                raise ValueError("episode内の教師actionがpadding扱いです")
            padded_count += 1
        else:
            if source_index >= len(episode_actions) or not np.array_equal(
                target_chunk[step], episode_actions[source_index],
            ):
                raise ValueError("学習action chunkが教師次周期末actionと一致しません")
            valid_count += 1
    return valid_count, padded_count


def _qpos(row: dict) -> np.ndarray:
    return np.array([float(row[f"{name}_qpos"]) for name in JOINT_NAMES], dtype=np.float64)


def _action_metrics(
    errors: np.ndarray,
    pads: np.ndarray,
    indices: np.ndarray,
    applied_errors: np.ndarray,
    clip_corrections: np.ndarray,
) -> dict:
    if len(indices) == 0:
        return {
            "first_action_raw": summarize_errors(np.empty((0, 6))),
            "first_action_bounded_estimate": summarize_errors(np.empty((0, 6))),
            "first_action_safety_correction": summarize_errors(np.empty((0, 6))),
            "whole_valid_chunk": summarize_errors(np.empty((0, 6))),
        }
    valid = ~pads[indices]
    return {
        "first_action_raw": summarize_errors(errors[indices, 0, :]),
        "first_action_bounded_estimate": summarize_errors(applied_errors[indices]),
        "first_action_safety_correction": summarize_errors(clip_corrections[indices]),
        "whole_valid_chunk": summarize_errors(errors[indices][valid]),
    }


def _load_teacher_episodes(study_path: Path, dataset_path: Path) -> list[dict]:
    collection_path = study_path.parent / "collection" / "collection.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if Path(collection["dataset"]).resolve() != dataset_path.resolve():
        raise ValueError("教師コレクションと学習データのパスが一致しません")
    if collection.get("action_contract", {}).get("version") != ENDPOINT_MODE:
        raise ValueError("教師コレクションがendpoint_linear_v2ではありません")
    episodes = collection.get("episodes", [])
    if len(episodes) != 12:
        raise ValueError("事前固定した12教師episodeが必要です")
    loaded = []
    for episode_index, episode in enumerate(episodes):
        source = Path(episode["source"])
        if episode.get("frames") != 450 or source.name != f"train-{episode_index:02d}-endpoint":
            raise ValueError("教師episodeの順序または450 frame契約が一致しません")
        replay, endpoint_actions = load_endpoint_replay(source)
        if endpoint_actions.shape != (450, 6) or len(replay.rows) != 22501:
            raise ValueError("教師episodeの軌道形状が一致しません")
        loaded.append({
            "source": source,
            "replay": replay,
            "endpoint_actions": endpoint_actions,
            "source_sha256": {
                name: _sha256(source / name) for name in ("replay.json", "trajectory.csv")
            },
        })
    return loaded


def _make_observation(sample: dict) -> dict:
    return {
        "observation.state": sample["observation.state"],
        "observation.images.camera1": sample["observation.images.camera1"],
        "observation.images.camera2": sample["observation.images.camera2"],
        "task": sample["task"],
    }


def _preprocess_batch(samples: list[dict], preprocessor, torch) -> dict:
    processed = [preprocessor(_make_observation(sample)) for sample in samples]
    tensor_keys = {
        key for key, value in processed[0].items() if isinstance(value, torch.Tensor)
    }
    if not tensor_keys:
        raise ValueError("SmolVLA前処理結果にtensor入力がありません")
    for item in processed:
        keys = {key for key, value in item.items() if isinstance(value, torch.Tensor)}
        if keys != tensor_keys:
            raise ValueError("同一batch内でSmolVLA前処理のtensorキーが変化しました")
    return {key: torch.cat([item[key] for item in processed], dim=0) for key in tensor_keys}


def _write_action_predictions(
    path: Path,
    policy,
    preprocessor,
    postprocessor,
    dataset,
    teachers: list[dict],
    actuator_ranges: np.ndarray,
    batch_size: int,
) -> tuple:
    import torch

    sample_count = len(dataset)
    predictions = np.empty((sample_count, CHUNK_SIZE, 6), dtype=np.float32)
    labels = np.empty_like(predictions)
    pads = np.empty((sample_count, CHUNK_SIZE), dtype=bool)
    episode_ids = np.empty(sample_count, dtype=np.int64)
    times = np.empty(sample_count, dtype=np.float64)
    previous_teacher = np.empty((sample_count, 6), dtype=np.float64)
    phases = [""] * sample_count
    applied = np.empty((sample_count, 6), dtype=np.float64)
    valid_label_count = 0
    padded_label_count = 0

    fieldnames = [
        "dataset_index", "episode_index", "frame_index", "dataset_timestamp_s",
        "observation_time_s",
        "phase", "chunk_step", "target_time_s", "action_is_pad",
        "selected_by_select_action", "teacher_previous_action_rad",
        "teacher_action_rad", "predicted_action_rad", "prediction_error_rad",
        "bounded_applied_action_rad", "safety_correction_rad",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        torch.manual_seed(DIAGNOSTIC_SEED)
        policy.eval()
        policy.reset()
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            samples = [dataset[index] for index in range(start, stop)]
            preprocessed = _preprocess_batch(samples, preprocessor, torch)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted = postprocessor(policy.predict_action_chunk(preprocessed))
            predicted_np = predicted.float().cpu().numpy()
            if predicted_np.shape != (len(samples), CHUNK_SIZE, 6):
                raise ValueError(
                    f"SmolVLA action chunk shapeが不正です: {predicted_np.shape}"
                )
            if not np.isfinite(predicted_np).all():
                raise ValueError(f"非有限actionが出力されました: sample {start}")

            for local_index, sample in enumerate(samples):
                global_index = start + local_index
                episode_id = int(sample["episode_index"])
                frame_id = int(sample["frame_index"])
                if episode_id < 0 or episode_id >= len(teachers) or not 0 <= frame_id < 450:
                    raise ValueError("datasetのepisode/frame indexが教師データ範囲外です")
                teacher = teachers[episode_id]["replay"]
                physics_index = frame_id * 50
                dataset_time = float(sample["timestamp"])
                sample_time = _aligned_sample_time(dataset_time, frame_id)
                target_chunk = sample["action"].detach().cpu().numpy()
                pad_chunk = sample["action_is_pad"].detach().cpu().numpy().astype(bool)
                if target_chunk.shape != (CHUNK_SIZE, 6) or pad_chunk.shape != (CHUNK_SIZE,):
                    raise ValueError("教師action chunkの形状が不正です")
                if pad_chunk[0]:
                    raise ValueError("select_actionの先頭教師actionがpaddingです")
                valid_count, padded_count = _validate_action_chunk(
                    target_chunk,
                    pad_chunk,
                    teachers[episode_id]["endpoint_actions"],
                    frame_id,
                )
                valid_label_count += valid_count
                padded_label_count += padded_count
                target_previous = _target(teacher.rows[physics_index])
                phase = str(teacher.rows[physics_index]["phase"])
                sample_predictions = predicted_np[local_index]
                bounded = bounded_action(
                    sample_predictions[0].astype(np.float64),
                    target_previous,
                    actuator_ranges,
                )

                predictions[global_index] = sample_predictions
                labels[global_index] = target_chunk
                pads[global_index] = pad_chunk
                episode_ids[global_index] = episode_id
                times[global_index] = sample_time
                previous_teacher[global_index] = target_previous
                phases[global_index] = phase
                applied[global_index] = bounded

                for chunk_step in range(CHUNK_SIZE):
                    teacher_action = target_chunk[chunk_step]
                    predicted_action = sample_predictions[chunk_step]
                    selected = chunk_step == 0
                    writer.writerow({
                        "dataset_index": global_index,
                        "episode_index": episode_id,
                        "frame_index": frame_id,
                        "dataset_timestamp_s": f"{dataset_time:.9f}",
                        "observation_time_s": f"{sample_time:.6f}",
                        "phase": phase,
                        "chunk_step": chunk_step,
                        "target_time_s": f"{sample_time + (chunk_step + 1) / FPS:.6f}",
                        "action_is_pad": bool(pad_chunk[chunk_step]),
                        "selected_by_select_action": selected,
                        "teacher_previous_action_rad": json.dumps(target_previous.tolist()),
                        "teacher_action_rad": json.dumps(teacher_action.tolist()),
                        "predicted_action_rad": json.dumps(predicted_action.tolist()),
                        "prediction_error_rad": json.dumps((predicted_action - teacher_action).tolist()),
                        "bounded_applied_action_rad": json.dumps(bounded.tolist()) if selected else "",
                        "safety_correction_rad": json.dumps((bounded - sample_predictions[0]).tolist())
                        if selected else "",
                    })
            if stop % 100 == 0 or stop == sample_count:
                print(f"オフラインaction推論: {stop}/{sample_count} frames", flush=True)

    return (
        {
            "sample_count": sample_count, "batch_size": batch_size,
            "verified_teacher_action_targets": valid_label_count,
            "excluded_padding_targets": padded_label_count,
        },
        phases, predictions, labels, pads, applied, previous_teacher, times, episode_ids,
    )


def _summarize_action_fit(
    predictions: np.ndarray,
    labels: np.ndarray,
    pads: np.ndarray,
    applied: np.ndarray,
    previous_teacher: np.ndarray,
    episode_ids: np.ndarray,
    times: np.ndarray,
    phases: list[str],
    episode_names: list[str],
) -> dict:
    errors = predictions.astype(np.float64) - labels.astype(np.float64)
    applied_errors = applied - labels[:, 0, :].astype(np.float64)
    corrections = applied - predictions[:, 0, :].astype(np.float64)
    all_indices = np.arange(len(episode_ids))
    result = {
        "frames": len(episode_ids),
        "first_action_is_select_action_output": True,
        "action_chunk_size": CHUNK_SIZE,
        "first_action_raw": summarize_errors(errors[:, 0, :]),
        "first_action_bounded_estimate": summarize_errors(applied_errors),
        "first_action_safety_correction": summarize_errors(corrections),
        "whole_valid_chunk": summarize_errors(errors[~pads]),
        "valid_action_count": int((~pads).sum()),
        "padded_action_count_excluded_from_metrics": int(pads.sum()),
        "per_chunk_step": {
            str(step): summarize_errors(errors[~pads[:, step], step, :])
            for step in range(CHUNK_SIZE)
        },
        "by_episode": {},
        "by_phase": {},
        "by_observation_time_s": {},
        "safety_clip": {
            "frames_with_any_correction": int(np.any(np.abs(corrections) > 1e-12, axis=1).sum()),
            "max_abs_correction_rad": float(np.abs(corrections).max()),
            "scope": "bounded_actionによる関節余裕・1周期変化量のみ。接触依存open抑止は推定しない。",
        },
    }
    for episode_id, name in enumerate(episode_names):
        indices = all_indices[episode_ids == episode_id]
        result["by_episode"][name] = _action_metrics(
            errors, pads, indices, applied_errors, corrections,
        )
    for phase in sorted(set(phases)):
        indices = all_indices[np.array(phases) == phase]
        result["by_phase"][phase] = _action_metrics(
            errors, pads, indices, applied_errors, corrections,
        )
    windows = [
        (0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 45.0),
    ]
    for start, end in windows:
        indices = all_indices[(times >= start) & (times < end)]
        result["by_observation_time_s"][f"{start:g}-{end:g}"] = _action_metrics(
            errors, pads, indices, applied_errors, corrections,
        )
    return result


def _write_rollout_comparison(
    directory: Path,
    evaluation_path: Path,
    evaluation: dict,
    teacher,
) -> dict:
    trajectory_path = evaluation_path / "trajectory.csv"
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        policy_rows = list(csv.DictReader(stream))
    if len(policy_rows) != int(evaluation["physics_steps"]) + 1:
        raise ValueError("既存評価のCSV行数とphysics_stepsが一致しません")

    state_fields = [
        "time_s", "policy_phase", "teacher_phase", "unexpected_contacts",
        "policy_joint_qpos_rad", "teacher_joint_qpos_rad", "joint_state_error_rad",
        "policy_joint_target_rad", "teacher_joint_target_rad", "joint_target_error_rad",
        "policy_dice_xyz_m", "teacher_dice_xyz_m", "dice_position_error_m",
        "dice_orientation_error_rad",
    ]
    joint_state_errors = []
    joint_target_errors = []
    dice_position_errors = []
    orientation_errors = []
    contact_rows = []
    first_state_deviation = None
    with (directory / "rollout_divergence.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=state_fields)
        writer.writeheader()
        for index, row in enumerate(policy_rows):
            policy_time = float(row["time_s"])
            teacher_index = round(policy_time / PHYSICS_TIMESTEP_S)
            if teacher_index >= len(teacher.rows) or not np.isclose(
                policy_time, teacher_index * PHYSICS_TIMESTEP_S, atol=1e-9, rtol=0,
            ):
                raise ValueError("既存SmolVLA軌道を教師500Hz時刻へ整列できません")
            teacher_row = teacher.rows[teacher_index]
            policy_qpos = _qpos(row)
            teacher_qpos = teacher.qpos[teacher_index, :6]
            state_error = policy_qpos - teacher_qpos
            policy_target = _target(row)
            teacher_target = _target(teacher_row)
            target_error = policy_target - teacher_target
            policy_xyz = np.array([float(row[f"dice_{axis}_m"]) for axis in "xyz"])
            teacher_xyz = np.array([float(teacher_row[f"dice_{axis}_m"]) for axis in "xyz"])
            position_error = policy_xyz - teacher_xyz
            policy_quat = np.array([float(row[f"dice_quat_{axis}"]) for axis in "wxyz"])
            teacher_quat = teacher.qpos[teacher_index, 9:13]
            orientation_error = _quat_angle_rad(policy_quat, teacher_quat)
            unexpected = int(row["unexpected_contacts"])
            joint_state_errors.append(state_error)
            joint_target_errors.append(target_error)
            dice_position_errors.append(position_error)
            orientation_errors.append(orientation_error)
            if first_state_deviation is None and np.max(np.abs(state_error)) > 1e-9:
                first_state_deviation = {
                    "time_s": policy_time,
                    "max_abs_joint_error_rad": float(np.max(np.abs(state_error))),
                    "joint_errors_rad": state_error.tolist(),
                }
            if unexpected and not contact_rows:
                contact_rows.append({
                    "time_s": policy_time, "phase": row["phase"],
                    "unexpected_contact_count": unexpected,
                })
            writer.writerow({
                "time_s": f"{policy_time:.6f}",
                "policy_phase": row["phase"],
                "teacher_phase": teacher_row["phase"],
                "unexpected_contacts": unexpected,
                "policy_joint_qpos_rad": json.dumps(policy_qpos.tolist()),
                "teacher_joint_qpos_rad": json.dumps(teacher_qpos.tolist()),
                "joint_state_error_rad": json.dumps(state_error.tolist()),
                "policy_joint_target_rad": json.dumps(policy_target.tolist()),
                "teacher_joint_target_rad": json.dumps(teacher_target.tolist()),
                "joint_target_error_rad": json.dumps(target_error.tolist()),
                "policy_dice_xyz_m": json.dumps(policy_xyz.tolist()),
                "teacher_dice_xyz_m": json.dumps(teacher_xyz.tolist()),
                "dice_position_error_m": json.dumps(position_error.tolist()),
                "dice_orientation_error_rad": f"{orientation_error:.12g}",
            })

    controls = evaluation.get("controls", [])
    control_errors = []
    applied_errors = []
    correction_errors = []
    first_control_deviation = None
    control_fields = [
        "control_time_s", "target_time_s", "teacher_target_row", "teacher_phase",
        "blocked_open_guard", "policy_raw_action_rad", "policy_applied_action_rad",
        "teacher_endpoint_action_rad", "raw_action_error_rad", "applied_action_error_rad",
        "safety_correction_rad",
    ]
    with (directory / "rollout_controls.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=control_fields)
        writer.writeheader()
        for cycle, control in enumerate(controls):
            start_index = round(float(control["time_s"]) / PHYSICS_TIMESTEP_S)
            target_index = start_index + round((1 / FPS) / PHYSICS_TIMESTEP_S)
            if target_index >= len(teacher.rows):
                raise ValueError("教師軌道にSmolVLA末端指令の比較時刻がありません")
            if not np.isclose(
                float(control["time_s"]), start_index * PHYSICS_TIMESTEP_S,
                atol=1e-9, rtol=0,
            ):
                raise ValueError("SmolVLA control時刻を教師周期へ整列できません")
            teacher_endpoint = _target(teacher.rows[target_index])
            raw = np.asarray(control["raw_action_rad"], dtype=np.float64)
            bounded = np.asarray(control["applied_action_rad"], dtype=np.float64)
            if raw.shape != (6,) or bounded.shape != (6,) or not np.isfinite(raw).all():
                raise ValueError(f"既存rolloutのcontrol actionが不正です: {cycle}")
            raw_error = raw - teacher_endpoint
            applied_error = bounded - teacher_endpoint
            correction = bounded - raw
            control_errors.append(raw_error)
            applied_errors.append(applied_error)
            correction_errors.append(correction)
            if first_control_deviation is None and np.max(np.abs(raw_error)) > 1e-9:
                first_control_deviation = {
                    "cycle": cycle,
                    "control_time_s": float(control["time_s"]),
                    "target_time_s": target_index * PHYSICS_TIMESTEP_S,
                    "teacher_endpoint_row": target_index,
                    "max_abs_raw_action_error_rad": float(np.max(np.abs(raw_error))),
                    "raw_action_error_rad": raw_error.tolist(),
                    "applied_action_error_rad": applied_error.tolist(),
                }
            writer.writerow({
                "control_time_s": f"{float(control['time_s']):.6f}",
                "target_time_s": f"{target_index * PHYSICS_TIMESTEP_S:.6f}",
                "teacher_target_row": target_index,
                "teacher_phase": teacher.rows[start_index]["phase"],
                "blocked_open_guard": bool(control["blocked_open"]),
                "policy_raw_action_rad": json.dumps(raw.tolist()),
                "policy_applied_action_rad": json.dumps(bounded.tolist()),
                "teacher_endpoint_action_rad": json.dumps(teacher_endpoint.tolist()),
                "raw_action_error_rad": json.dumps(raw_error.tolist()),
                "applied_action_error_rad": json.dumps(applied_error.tolist()),
                "safety_correction_rad": json.dumps(correction.tolist()),
            })

    state_errors = np.asarray(joint_state_errors)
    target_errors = np.asarray(joint_target_errors)
    position_errors = np.asarray(dice_position_errors)
    control_errors = np.asarray(control_errors).reshape((-1, 6))
    applied_errors = np.asarray(applied_errors).reshape((-1, 6))
    correction_errors = np.asarray(correction_errors).reshape((-1, 6))
    contact = contact_rows[0] if contact_rows else None
    return {
        "evaluation_duration_s": evaluation["duration_s"],
        "evaluation_stop_reason": evaluation["failure_reasons"],
        "evaluation_completed": evaluation["completed"],
        "initial_joint_state_matches_teacher": bool(np.array_equal(state_errors[0], np.zeros(6))),
        "first_control_divergence": first_control_deviation,
        "first_joint_state_divergence": first_state_deviation,
        "first_unexpected_contact": contact,
        "control_endpoint_raw_vs_teacher": summarize_errors(control_errors),
        "control_endpoint_applied_vs_teacher": summarize_errors(applied_errors),
        "control_endpoint_safety_correction": summarize_errors(correction_errors),
        "physics_step_joint_state_vs_teacher": summarize_errors(state_errors),
        "physics_step_joint_target_vs_teacher": summarize_errors(target_errors),
        "dice_position_error_m": {
            "mae_m": float(np.abs(position_errors).mean()),
            "max_abs_m": float(np.abs(position_errors).max()),
            "final_xyz_error_m": position_errors[-1].tolist(),
        },
        "dice_orientation_error_rad": {
            "mean_rad": float(np.mean(orientation_errors)),
            "max_rad": float(np.max(orientation_errors)),
            "final_rad": float(orientation_errors[-1]),
        },
        "row_count": len(policy_rows),
        "control_count": len(controls),
        "contact_notice": "既存CSVのunexpected_contacts列だけを集計。接触形状・閾値は変更していない。",
    }


def run_diagnostics(
    directory: Path,
    study_path: Path,
    training_path: Path,
    evaluation_path: Path,
    batch_size: int = 4,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if batch_size < 1:
        raise ValueError("batch sizeは1以上である必要があります")
    study, training, teacher_path, central_teacher, offset = load_training_position(
        study_path, training_path,
    )
    if offset != [0.0, 0.0]:
        raise ValueError("中央学習位置(0,0)mmの診断ではありません")
    evaluation_file = evaluation_path / "evaluation.json"
    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    settings_file = evaluation_path.parent / "settings.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    if (
        Path(settings["training"]).resolve() != training_path.resolve()
        or Path(settings["study"]).resolve() != study_path.resolve()
        or evaluation.get("evaluation_type") != "training_position_diagnostic"
        or evaluation.get("requested_face") != 2
    ):
        raise ValueError("既存ロールアウトが指定した中央学習位置診断に対応しません")
    if not np.array_equal(np.asarray(evaluation["initial_qpos"]), central_teacher.qpos[0]):
        raise ValueError("既存ロールアウトと中央教師の初期qposが一致しません")
    require_contract = training.get("action_contract")
    if require_contract is None or require_contract.get("version") != ENDPOINT_MODE:
        raise ValueError("継続後checkpointのendpoint_linear_v2契約がありません")

    dataset_path = Path(training["dataset"])
    teachers = _load_teacher_episodes(study_path, dataset_path)
    dataset = LeRobotDataset(
        training["repo_id"],
        root=dataset_path,
        delta_timestamps={"action": [index / FPS for index in range(CHUNK_SIZE)]},
    )
    if len(dataset) != 5400 or dataset.meta.total_episodes != len(teachers):
        raise ValueError("事前固定した5,400 frames・12 episodesと一致しません")
    policy, preprocessor, postprocessor = load_policy(
        training["models"], dataset, Path(training["checkpoint"]),
        dataset_stats=training_normalization_stats(training, dataset),
    )
    policy.eval()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    if model.nu != len(JOINT_NAMES) or not np.isclose(
        model.opt.timestep, PHYSICS_TIMESTEP_S, atol=1e-12, rtol=0,
    ):
        raise ValueError("安全clip評価用MuJoCo modelが既存action契約と一致しません")

    inference, phases, predictions, labels, pads, applied, previous_teacher, times, episode_ids = (
        _write_action_predictions(
            directory / "offline_action_predictions.csv",
            policy,
            preprocessor,
            postprocessor,
            dataset,
            teachers,
            model.actuator_ctrlrange.copy(),
            batch_size,
        )
    )
    action_fit = _summarize_action_fit(
        predictions, labels, pads, applied, previous_teacher,
        episode_ids, times, phases, [item["source"].name for item in teachers],
    )
    rollout = _write_rollout_comparison(directory, evaluation_path, evaluation, central_teacher)

    checkpoint_files = checkpoint_fingerprint(Path(training["checkpoint"]))
    collection_path = study_path.parent / "collection" / "collection.json"
    summary = {
        "status": "complete",
        "diagnostic_only": True,
        "physics_rollout_executed": False,
        "training_performed": False,
        "uploaded": False,
        "seed": DIAGNOSTIC_SEED,
        "inference": {
            **inference,
            "mode": "eval/inference_mode/autocast_bfloat16",
            "action_contract": ENDPOINT_MODE,
            "first_action_semantics": "postprocess(predict_action_chunk(...))[0]。n_action_steps=1なのでselect_actionが返すaction。",
            "whole_chunk_separately_reported": True,
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "frame_count": len(dataset),
            "episode_count": dataset.meta.total_episodes,
            "fingerprint": dataset_fingerprint(dataset_path),
            "collection_sha256": _sha256(collection_path),
            "episode_sources": [
                {"path": str(item["source"].resolve()), "sha256": item["source_sha256"]}
                for item in teachers
            ],
        },
        "checkpoint": {
            "path": str(Path(training["checkpoint"]).resolve()),
            "files_sha256": checkpoint_files,
            "aggregate_sha256": _aggregate_fingerprint(checkpoint_files),
        },
        "contract": {
            "value": require_contract,
            "sha256": _sha256(dataset_path / "action_contract.json"),
        },
        "inputs_sha256": {
            "study": _sha256(study_path),
            "training": _sha256(training_path),
            "evaluation_settings": _sha256(settings_file),
            "evaluation": _sha256(evaluation_file),
            "evaluation_trajectory": _sha256(evaluation_path / "trajectory.csv"),
        },
        "evaluation_reference": {
            "path": str(evaluation_path.resolve()),
            "offset_mm": offset,
            "initial_qpos": evaluation["initial_qpos"],
            "failure_reasons": evaluation["failure_reasons"],
            "passed": evaluation["passed"],
        },
        "action_fit": action_fit,
        "rollout_divergence": rollout,
        "interpretation_limits": [
            "training-set action適合度は汎化性能・物理タスク成功を意味しない。",
            "bounded_actionのoffline適用値は各frameで前回教師指令を使う推定で、連続したSmolVLA閉ループではない。",
            "接触依存open抑止は、状態に対応する接触窓を再構成しないためoffline action fitには含めない。",
            "rollout比較は既存CSV/JSONだけを時刻整列した。物理シミュレーションを再実行していない。",
        ],
        "study_split": study["split"],
    }
    write_json(directory / "diagnostics.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SmolVLA training action fitと既存失敗rolloutをオフライン診断",
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/offline-diagnostics"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        result = run_diagnostics(
            directory, args.study, args.training, args.evaluation, args.batch_size,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        write_json(directory / "error.json", {
            "status": "error", "error_type": type(error).__name__, "message": str(error),
        })
        print(f"オフライン診断エラー: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"オフライン診断保存先: {directory.resolve()}", flush=True)
    print(
        "初動action MAE: "
        f"{result['action_fit']['first_action_raw']['mae_rad']:.6g} rad; "
        f"最初の指令分岐: {result['rollout_divergence']['first_control_divergence']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
