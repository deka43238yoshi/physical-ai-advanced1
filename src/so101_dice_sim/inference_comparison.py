import argparse
from contextlib import contextmanager, ExitStack
import csv
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np

from .action_contract import ENDPOINT_MODE, dataset_fingerprint, require_training_contract
from .contact_diagnostics import _policy_pose, _read_policy_trajectory
from .endpoint_collection import load_endpoint_replay
from .learning import (
    FPS, TASK, bounded_action, checkpoint_fingerprint, load_policy,
    render_observation, training_normalization_stats,
)
from .offline_diagnostics import (
    _aligned_sample_time, _make_observation, _target, _validate_action_chunk,
    summarize_errors,
)
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES, ROOT, verify_assets
from .training_position_evaluation import load_training_position


EPISODES = (4, 1, 11)
FRAMES = (*range(31), 50, 100, 150, 200, 250, 300, 350, 400, 449)
DRAWS = 4
SEED_BASE = 106
ERROR_REFERENCE_RAD = .05
CURRENT_METHODS = ("default10", "steps50", "mean4", "zero_noise")
CSV_FIELDS = (
    "candidate", "source", "group", "episode", "frame", "time_s", "phase",
    "draw", "seeds", "noise_sha256", "steps", "action_rad", "target_rad",
    "previous_command_rad", "bounded_estimate_rad", "error_rad", "policy_call_ms",
)


def noise_seed(episode: int, frame: int, draw: int) -> int:
    if (
        any(isinstance(value, bool) or not isinstance(value, int)
            for value in (episode, frame, draw))
        or episode not in EPISODES or frame not in FRAMES or not 0 <= draw < DRAWS
    ):
        raise ValueError("事前固定したepisode/frame/drawの範囲外です")
    return SEED_BASE + (episode * 450 + frame) * DRAWS + draw


def mean_four(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != (DRAWS, 6) or not np.isfinite(values).all():
        raise ValueError("4回平均には有限な4x6関節actionが必要です")
    return values.mean(axis=0)


def require_initial_match(action: np.ndarray, saved: np.ndarray) -> float:
    values, expected = np.asarray(action), np.asarray(saved)
    if (
        values.shape != (6,) or expected.shape != (6,)
        or not np.isfinite(values).all() or not np.isfinite(expected).all()
    ):
        raise ValueError("初期action照合には有限な6関節値が必要です")
    difference = float(np.max(np.abs(values - expected)))
    if difference > 1e-6:
        raise ValueError(f"保存済み初期actionを再現できません: 最大差{difference} rad")
    return difference


def require_reference_coverage(training: dict) -> None:
    indices = [
        index for batch in training["losses"] for index in batch["sample_indices"]
    ]
    if (
        training.get("samples_seen") != 5400 or len(indices) != 5400
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or sorted(indices) != list(range(5400))
        or training.get("normalization_stats") is not None
    ):
        raise ValueError("参考checkpointは全5,400 frameを学習した全dataset統計の記録が必要です")


@contextmanager
def sampling_steps(policy, steps: int):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("積分回数は正の整数が必要です")
    config = policy.model.config
    original = config.num_steps
    try:
        config.num_steps = steps
        yield
    finally:
        config.num_steps = original


@contextmanager
def forbid_physics_steps():
    def reject(*args, **kwargs):
        raise RuntimeError("オフライン比較中のMuJoCo物理stepは禁止です")

    with ExitStack() as stack:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            stack.enter_context(patch.object(mujoco, name, reject))
        yield


def control_indices(controls: list[dict], row_count: int) -> list[int]:
    if len(controls) != 31:
        raise ValueError("保存済み失敗評価の31 controlが必要です")
    indices = []
    for frame, control in enumerate(controls):
        timestamp = float(control["time_s"])
        if not np.isfinite(timestamp) or not np.isclose(
            timestamp, frame / FPS, atol=1e-9, rtol=0,
        ):
            raise ValueError("保存controlの時刻が10 Hz観測時刻と一致しません")
        for key in ("raw_action_rad", "applied_action_rad"):
            action = np.asarray(control[key], dtype=np.float64)
            if action.shape != (6,) or not np.isfinite(action).all():
                raise ValueError("保存controlのactionが不正です")
        index = frame * 50
        if index >= row_count:
            raise ValueError("保存controlの観測がtrajectory範囲外です")
        indices.append(index)
    return indices


def action_metrics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("候補比較の集計対象が空です")
    actions = np.asarray([row["action_rad"] for row in records], dtype=np.float64)
    targets = np.asarray([row["target_rad"] for row in records], dtype=np.float64)
    bounded = np.asarray([row["bounded_estimate_rad"] for row in records], dtype=np.float64)
    milliseconds = np.asarray([row["policy_call_ms"] for row in records], dtype=np.float64)
    if (
        actions.shape != (len(records), 6) or targets.shape != actions.shape
        or bounded.shape != actions.shape or not np.isfinite(bounded).all()
        or milliseconds.shape != (len(records),) or not np.isfinite(milliseconds).all()
        or np.any(milliseconds < 0)
    ):
        raise ValueError("候補のactionまたは計測時間が不正です")
    errors = actions - targets
    return {
        "observation_count": len({(row["source"], row["episode"], row["frame"])
                                  for row in records}),
        "raw": summarize_errors(errors),
        "bounded_estimate": summarize_errors(bounded - targets),
        "all_joints_within_0_05_rad_fraction": float(
            np.all(np.abs(errors) <= ERROR_REFERENCE_RAD, axis=1).mean(),
        ),
        "clipped_decision_fraction": float(
            np.any(np.abs(actions - bounded) > 1e-12, axis=1).mean(),
        ),
        "mean_signed_error_rad": errors.mean(axis=0).tolist(),
        "policy_call_ms": {
            "median": float(np.median(milliseconds)),
            "p95": float(np.quantile(milliseconds, .95)),
        },
    }


def paired_effect(records: list[dict]) -> dict:
    teachers = {
        (row["frame"], row["draw"]): row for row in records
        if row["source"] == "teacher" and row["episode"] == 4 and row["frame"] <= 30
    }
    differences = []
    for row in records:
        if row["source"] != "saved_policy":
            continue
        teacher = teachers.get((row["frame"], row["draw"]))
        if teacher is None or any(
            row[key] != teacher[key] for key in ("seeds", "noise_sha256", "steps")
        ):
            raise ValueError("教師・保存方策ペアのnoiseまたは積分設定が一致しません")
        differences.append(np.asarray(row["action_rad"]) - teacher["action_rad"])
    if not differences or len(differences) != len(teachers):
        raise ValueError("教師・保存方策ペアのサンプル数が一致しません")
    return summarize_errors(np.asarray(differences))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(files: dict[str, Path], trainings: dict[str, dict]) -> dict:
    return {
        "files_sha256": {name: _sha256(path) for name, path in files.items()},
        "dataset": dataset_fingerprint(Path(trainings["current"]["dataset"])),
        "checkpoints": {
            name: checkpoint_fingerprint(Path(training["checkpoint"]))
            for name, training in trainings.items()
        },
    }


def _observations(dataset, teachers: dict, policy_rows: list[dict], controls: list[dict]):
    import torch
    from .visualization import ReplayRenderer

    samples = []
    for episode in EPISODES:
        teacher, endpoints = teachers[episode]
        for frame in FRAMES:
            sample = dataset[episode * 450 + frame]
            if int(sample["episode_index"]) != episode or int(sample["frame_index"]) != frame:
                raise ValueError("datasetのepisode/frame順序が想定と一致しません")
            _aligned_sample_time(float(sample["timestamp"]), frame)
            _validate_action_chunk(
                sample["action"].cpu().numpy(), sample["action_is_pad"].cpu().numpy(),
                endpoints, frame,
            )
            if sample["task"] != TASK or not np.array_equal(
                sample["observation.state"].cpu().numpy(),
                teacher.qpos[frame * 50, :6].astype(np.float32),
            ):
                raise ValueError("datasetの状態または指示が教師と一致しません")
            split = "train_central" if episode == 4 else "internal_validation"
            samples.append({
                "source": "teacher", "group": f"{split}_{'early' if frame <= 30 else 'later'}",
                "episode": episode, "frame": frame, "time_s": frame / FPS,
                "phase": teacher.rows[frame * 50]["phase"],
                "observation": _make_observation(sample),
                "target_rad": endpoints[frame].astype(np.float64),
                "previous_command_rad": _target(teacher.rows[frame * 50]),
            })
    renderer = ReplayRenderer()
    try:
        if (renderer.model.nq, renderer.model.nu) != (13, 6) or not np.isclose(
            renderer.model.opt.timestep, .002, atol=1e-12, rtol=0,
        ):
            raise ValueError("描画modelが既存の物理契約と一致しません")
        ranges = renderer.model.actuator_ctrlrange.copy()
        for frame, index in enumerate(control_indices(controls, len(policy_rows))):
            row = policy_rows[index]
            pose = _policy_pose(row)
            observation = {
                key: torch.from_numpy(image).permute(2, 0, 1).float() / 255
                for key, image in render_observation(renderer, pose).items()
            }
            observation.update({
                "observation.state": torch.tensor(pose[:6], dtype=torch.float32), "task": TASK,
            })
            samples.append({
                "source": "saved_policy", "group": "saved_policy_early", "episode": 4,
                "frame": frame, "time_s": frame / FPS, "phase": row["phase"],
                "observation": observation,
                "target_rad": teachers[4][1][frame].astype(np.float64),
                "previous_command_rad": _target(row),
            })
        if renderer.data.time != 0:
            raise RuntimeError("保存姿勢の描画中に物理時刻が進行しました")
    finally:
        renderer.close()
    return samples, ranges


def _noise(policy, seed: int):
    import torch

    torch.manual_seed(seed)
    return policy.model.sample_noise(
        (1, policy.config.chunk_size, policy.config.max_action_dim), "cuda",
    )


def _predict(policy, postprocessor, processed: dict, noise, steps: int):
    import torch

    policy.reset()
    with sampling_steps(policy, steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action = postprocessor(policy.select_action(
                dict(processed), noise=None if noise is None else noise.clone(),
            )).squeeze(0).float().cpu().numpy()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError("推論actionは有限な6関節値である必要があります")
    return action.astype(np.float64), elapsed_ms


def _record(sample: dict, candidate: str, action: np.ndarray, draw: int,
            seeds: list[int], hashes: list[str], steps: int, elapsed_ms: float,
            ranges: np.ndarray) -> dict:
    return {
        **{key: sample[key] for key in (
            "source", "group", "episode", "frame", "time_s", "phase",
            "target_rad", "previous_command_rad",
        )},
        "candidate": candidate, "draw": draw, "seeds": seeds, "noise_sha256": hashes,
        "steps": steps, "action_rad": action,
        "bounded_estimate_rad": bounded_action(action, sample["previous_command_rad"], ranges),
        "error_rad": action - sample["target_rad"], "policy_call_ms": elapsed_ms,
    }


def _csv_record(record: dict) -> dict:
    return {
        key: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
        if isinstance(value, (np.ndarray, list)) else value
        for key, value in record.items()
    }


def _compare_checkpoint(label, training, dataset, samples, ranges, writer, saved_initial):
    import torch

    policy, pre, post = load_policy(
        training["models"], dataset, Path(training["checkpoint"]),
        dataset_stats=training_normalization_stats(training, dataset),
    )
    policy.eval()
    if (
        (policy.config.chunk_size, policy.config.n_action_steps, policy.config.n_obs_steps)
        != (10, 1, 1)
        or policy.model.config.num_steps != 10 or policy.config.max_action_dim != 32
    ):
        raise ValueError("checkpointの推論設定が固定比較の前提と一致しません")
    initial = pre(dict(samples[0]["observation"]))
    torch.manual_seed(42)
    implicit, _ = _predict(policy, post, initial, None, 10)
    explicit, _ = _predict(policy, post, initial, _noise(policy, 42), 10)
    calibration = {"implicit_vs_explicit_noise_max_abs_rad": require_initial_match(explicit, implicit)}
    if label == "current":
        calibration["saved_initial_max_abs_rad"] = require_initial_match(explicit, saved_initial)
        _predict(policy, post, initial, _noise(policy, 42), 50)

    records = []
    for index, sample in enumerate(samples):
        processed = pre(dict(sample["observation"]))
        seeds = [noise_seed(sample["episode"], sample["frame"], draw) for draw in range(DRAWS)]
        noises = [_noise(policy, seed) for seed in seeds]
        hashes = [hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest() for noise in noises]
        actions, timings = [], []
        added = []
        for draw, noise in enumerate(noises):
            action, elapsed = _predict(policy, post, processed, noise, 10)
            actions.append(action)
            timings.append(elapsed)
            added.append(_record(
                sample, f"{label}.default10", action, draw, [seeds[draw]], [hashes[draw]],
                10, elapsed, ranges,
            ))
        if label == "current":
            added.append(_record(
                sample, "current.mean4", mean_four(np.asarray(actions)), 0, seeds, hashes,
                10, sum(timings), ranges,
            ))
            for draw, noise in enumerate(noises):
                action, elapsed = _predict(policy, post, processed, noise, 50)
                added.append(_record(
                    sample, "current.steps50", action, draw, [seeds[draw]], [hashes[draw]],
                    50, elapsed, ranges,
                ))
            zero = torch.zeros_like(noises[0])
            action, elapsed = _predict(policy, post, processed, zero, 10)
            added.append(_record(
                sample, "current.zero_noise", action, 0, [],
                [hashlib.sha256(zero.cpu().numpy().tobytes()).hexdigest()], 10, elapsed, ranges,
            ))
        records.extend(added)
        writer.writerows(_csv_record(row) for row in added)
        if (index + 1) % 10 == 0 or index + 1 == len(samples):
            print(f"候補比較 {label}: {index + 1}/{len(samples)} observations", flush=True)
    return records, calibration


@forbid_physics_steps()
def run_comparison(directory: Path, study_path: Path, training_path: Path,
                   evaluation_path: Path, reference_training_path: Path) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _, training, central_path, central, offset = load_training_position(study_path, training_path)
    if (
        offset != [0.0, 0.0]
        or training.get("training_episode_indices") != [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        or training.get("validation_episode_indices") != [1, 11]
        or training.get("normalization_stats_source_episode_indices")
        != training["training_episode_indices"]
    ):
        raise ValueError("現在のcheckpointが固定internal validation学習に対応しません")
    reference = json.loads(reference_training_path.read_text(encoding="utf-8"))
    require_training_contract(reference, ENDPOINT_MODE)
    require_reference_coverage(reference)
    if (
        Path(reference["dataset"]).resolve() != Path(training["dataset"]).resolve()
        or reference["repo_id"] != training["repo_id"]
        or Path(reference["checkpoint"]).resolve() == Path(training["checkpoint"]).resolve()
    ):
        raise ValueError("参考checkpointが同じdatasetの別checkpointではありません")
    evaluation = json.loads((evaluation_path / "evaluation.json").read_text(encoding="utf-8"))
    settings_path = evaluation_path.parent / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if (
        Path(settings["training"]).resolve() != training_path.resolve()
        or Path(settings["study"]).resolve() != study_path.resolve()
        or settings["training_sha256"] != _sha256(training_path)
        or settings["study_sha256"] != _sha256(study_path)
        or evaluation.get("evaluation_type") != "training_position_diagnostic"
        or evaluation.get("requested_face") != 2 or evaluation.get("offset_mm") != [0.0, 0.0]
        or evaluation.get("passed") is not False or evaluation.get("completed") is not False
        or evaluation.get("seed") != 42
        or not np.array_equal(evaluation["initial_qpos"], central.qpos[0])
    ):
        raise ValueError("保存評価が指定checkpointの中央位置失敗評価と一致しません")
    rows = _read_policy_trajectory(evaluation_path / "trajectory.csv", evaluation)
    control_indices(evaluation["controls"], len(rows))
    dataset_path = Path(training["dataset"])
    collection_path = study_path.parent / "collection" / "collection.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if Path(collection["dataset"]).resolve() != dataset_path.resolve() or len(collection["episodes"]) != 12:
        raise ValueError("教師コレクションが固定12 episode datasetと一致しません")
    teachers = {}
    files = {
        "study": study_path, "training": training_path, "reference_training": reference_training_path,
        "evaluation": evaluation_path / "evaluation.json", "evaluation_settings": settings_path,
        "trajectory": evaluation_path / "trajectory.csv", "collection": collection_path,
        "dataset_contract": dataset_path / "action_contract.json",
        "scene": ROOT / "assets" / "so101_dice_scene.xml",
        "diagnostic_code": Path(__file__),
    }
    provenance = verify_assets(ROOT / "assets" / "robotstudio_so101")
    for item in provenance["files"]:
        files[f"robot_asset:{item['path']}"] = ROOT / "assets" / "robotstudio_so101" / item["path"]
    for episode in EPISODES:
        item = collection["episodes"][episode]
        source = Path(item["source"])
        if item["frames"] != 450 or source != study_path.parent / f"train-{episode:02d}-endpoint":
            raise ValueError("教師episodeの順序・経路・frame数が一致しません")
        teachers[episode] = load_endpoint_replay(source)
        for name in ("replay.json", "trajectory.csv"):
            files[f"teacher:{episode}:{name}"] = source / name
    if central_path.resolve() != (study_path.parent / "train-04-endpoint").resolve():
        raise ValueError("中央教師の経路が一致しません")
    trainings = {"current": training, "reference": reference}
    fingerprints = _inputs(files, trainings)
    dataset = LeRobotDataset(
        training["repo_id"], root=dataset_path,
        delta_timestamps={"action": [index / FPS for index in range(10)]},
    )
    if len(dataset) != 5400 or dataset.meta.total_episodes != 12:
        raise ValueError("固定5,400 frames・12 episodes datasetが必要です")
    samples, ranges = _observations(dataset, teachers, rows, evaluation["controls"])
    protocol = {
        "episodes": list(EPISODES), "frame_indices": list(FRAMES), "draws": DRAWS,
        "seed_rule": "106 + (episode * 450 + frame) * 4 + draw",
        "current_methods": list(CURRENT_METHODS), "reference_methods": ["default10"],
        "observation_count": len(samples), "saved_policy_state_count": 31,
        "error_reference_rad": ERROR_REFERENCE_RAD,
        "reference_checkpoint_is_not_independent_validation": True,
        "normalization_source": {"current": "train_episodes_only", "reference": "full_dataset"},
        "policy_inputs": "images, joint state and task only; targets, phases and times are not inputs",
        "paths": {label: str(Path(record["checkpoint"]).resolve())
                  for label, record in trainings.items()},
    }
    write_json(directory / "settings.json", {"protocol": protocol, "inputs": fingerprints})
    all_records, calibration = [], {}
    with (directory / "candidate_actions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for label, record in trainings.items():
            records, calibration[label] = _compare_checkpoint(
                label, record, dataset, samples, ranges, writer,
                np.asarray(evaluation["controls"][0]["raw_action_rad"], dtype=np.float64),
            )
            all_records.extend(records)
            stream.flush()
            gc.collect()
            torch.cuda.empty_cache()
    if _inputs(files, trainings) != fingerprints:
        raise RuntimeError("比較中に入力dataset・checkpoint・軌道・コード・資産が変更されました")
    summaries = {}
    for candidate in sorted({row["candidate"] for row in all_records}):
        records = [row for row in all_records if row["candidate"] == candidate]
        summaries[candidate] = {
            "groups": {
                group: action_metrics([row for row in records if row["group"] == group])
                for group in sorted({row["group"] for row in records})
            },
            "teacher_episodes": {
                str(episode): action_metrics([
                    row for row in records if row["source"] == "teacher" and row["episode"] == episode
                ]) for episode in EPISODES
            },
            "paired_observation_action_delta": paired_effect(records),
        }
    report = {
        "status": "complete", "diagnostic_only": True, "production_policy_changed": False,
        "training_performed": False, "physics_steps_executed": 0, "uploaded": False,
        "physics_step_guard": ["mj_step", "mj_step1", "mj_step2"],
        "official_held_out_positions_used": False, "inputs_unchanged": True,
        "protocol": protocol, "calibration": calibration, "candidates": summaries,
        "inputs": fingerprints,
        "runtime": {**{name: version(name) for name in ("torch", "mujoco", "lerobot")},
                    "gpu": torch.cuda.get_device_name(0)},
        "limits": [
            "各候補は同じ保存観測で比較。閉ループ・接触・物理成功は検証していない。",
            "内部validationは既にcheckpoint選択に使用済み。参考checkpointは全12 episode学習済み。",
            "後半は9 frame/episodeの疎な確認であり、全軌道評価ではない。",
            "保存方策状態の教師目標は同時刻の参照値であり、最適な回復actionの正解ではない。",
            "0.05 rad以内は教師適合の参考指標であり、安全性の合格条件ではない。",
            "bounded_estimateは保存直前指令からの単発clipのみ。別候補の連続実行や開指guardを再現しない。",
            "mean4は4回のrad出力をclip前に平均。異なる動作モードの平均が有効とは限らない。",
            "zero_noiseは通常のGaussian noise分布ではない診断用probe。実行設定として採用していない。",
            "所要時間はGPU同期したpolicy呼び出しのみ。描画・前処理を除き、実時間制御の保証ではない。",
        ],
    }
    write_json(directory / "comparison.json", report)
    with (directory / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "candidate", "group", "observations", "decisions", "mae_rad", "rmse_rad",
            "max_abs_rad", "all_joints_within_0_05_rad_fraction", "clipped_fraction",
            "bounded_mae_rad", "policy_call_median_ms", "policy_call_p95_ms",
        ])
        for candidate, result in summaries.items():
            for group, metric in result["groups"].items():
                writer.writerow([
                    candidate, group, metric["observation_count"], metric["raw"]["sample_count"],
                    metric["raw"]["mae_rad"], metric["raw"]["rmse_rad"], metric["raw"]["max_abs_rad"],
                    metric["all_joints_within_0_05_rad_fraction"], metric["clipped_decision_fraction"],
                    metric["bounded_estimate"]["mae_rad"], metric["policy_call_ms"]["median"],
                    metric["policy_call_ms"]["p95"],
                ])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="SmolVLA推論候補を保存観測だけで固定比較")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference-training", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/inference-comparisons"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        run_comparison(
            directory, args.study.resolve(), args.training.resolve(),
            args.evaluation.resolve(), args.reference_training.resolve(),
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        write_json(directory / "error.json", {
            "status": "error", "error_type": type(error).__name__, "message": str(error),
        })
        print(f"オフライン候補比較エラー: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"オフライン候補比較保存先: {directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
