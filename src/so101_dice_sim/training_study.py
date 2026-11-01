import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .action_contract import ENDPOINT_MODE, dataset_fingerprint, require_training_contract
from .endpoint_collection import load_endpoint_replay
from .learning import (
    FPS,
    bounded_action,
    checkpoint_fingerprint,
    load_policy,
    training_normalization_stats,
)
from .learning_study import split_positions
from .offline_diagnostics import _preprocess_batch, summarize_errors
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES


VALIDATION_EPISODES = (1, 11)
CENTRAL_EPISODE = 4
MAX_EPOCHS = 3
BATCH_SIZE = 4
SEED = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint_json(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_episode_split(dataset, study: dict) -> tuple[dict, list[int], list[int]]:
    expected = json.loads(json.dumps(split_positions()))
    if study.get("split") != expected:
        raise ValueError("studyの事前固定位置分割が一致しません")
    episode_rows = list(dataset.meta.episodes)
    if len(episode_rows) != 12 or dataset.num_episodes != 12 or dataset.num_frames != 5400:
        raise ValueError("12 episode・5,400 frameの既存データが必要です")
    offsets = expected["train_offsets_mm"]
    episode_records = []
    train_sample_indices = []
    validation_sample_indices = []
    for episode_index, row in enumerate(episode_rows):
        begin = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        if (
            int(row["episode_index"]) != episode_index or length != 450
            or begin != episode_index * 450 or end - begin != length
        ):
            raise ValueError("episodeの並びまたはframe範囲が事前固定splitと一致しません")
        destination = validation_sample_indices if episode_index in VALIDATION_EPISODES else train_sample_indices
        destination.extend(range(begin, end))
        episode_records.append({
            "episode_index": episode_index,
            "dataset_frame_range": [begin, end],
            "offset_mm": offsets[episode_index],
            "role": "validation" if episode_index in VALIDATION_EPISODES else "train",
        })
    if CENTRAL_EPISODE in VALIDATION_EPISODES:
        raise ValueError("中央物理評価位置は学習側に残す必要があります")
    if len(train_sample_indices) != 4500 or len(validation_sample_indices) != 900:
        raise ValueError("固定splitのtrain/validation sample数が不一致です")
    manifest = {
        "version": 1,
        "target_face": 2,
        "initial_face_up": 1,
        "dataset_episode_count": 12,
        "dataset_frame_count": 5400,
        "validation_episode_indices": list(VALIDATION_EPISODES),
        "train_episode_indices": [
            index for index in range(12) if index not in VALIDATION_EPISODES
        ],
        "central_physics_evaluation_episode": CENTRAL_EPISODE,
        "episodes": episode_records,
        "official_held_out_positions_used": False,
        "validation_locked_before_training": True,
    }
    return manifest, train_sample_indices, validation_sample_indices


def _training_only_stats(dataset_path: Path, train_episode_indices: list[int]) -> dict:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import get_feature_stats

    files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("学習データのParquetがありません")
    tables = [
        pq.read_table(
            path,
            columns=["index", "episode_index", "observation.state", "action"],
        )
        for path in files
    ]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    mask = pc.is_in(
        table["episode_index"],
        value_set=pa.array(train_episode_indices, type=table["episode_index"].type),
    )
    selected = table.filter(mask).sort_by("index")
    if len(selected) != 4500:
        raise ValueError("学習側episodeから抽出したframe数が不正です")
    selected_episode_ids, episode_counts = np.unique(
        selected["episode_index"].to_numpy(), return_counts=True,
    )
    selected_indices = selected["index"].to_numpy()
    if (
        not np.array_equal(selected_episode_ids, np.asarray(train_episode_indices))
        or not np.array_equal(episode_counts, np.full(len(train_episode_indices), 450))
        or len(np.unique(selected_indices)) != 4500
    ):
        raise ValueError("Parquetのepisode selectionが固定splitと一致しません")

    result = {}
    for key in ("observation.state", "action"):
        values = np.asarray(selected[key].combine_chunks().values.to_numpy(), dtype=np.float32)
        if values.shape != (4500 * len(JOINT_NAMES),):
            raise ValueError(f"{key} Parquet列の要素数が一致しません")
        values = values.reshape(4500, len(JOINT_NAMES))
        if not np.isfinite(values).all():
            raise ValueError(f"{key}に非有限値があります")
        stats = get_feature_stats(values, axis=0, keepdims=False)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0):
            raise ValueError(f"{key}のtrain-only normalization統計が不正です")
        result[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": int(stats["count"][0]),
        }
    return result


def _sample_batches(indices: list[int], batch_size: int, generator: np.random.Generator):
    order = generator.permutation(indices).tolist()
    return [
        order[start:start + batch_size]
        for start in range(0, len(order), batch_size)
    ]


def _evaluate_indices(policy, preprocessor, postprocessor, dataset, indices, teachers, batch_size):
    import torch

    episode_names = sorted({str(index // 450) for index in indices})
    errors_by_episode = {name: [] for name in episode_names}
    chunk_errors_by_episode = {name: [] for name in episode_names}
    pads_by_episode = {name: [] for name in episode_names}
    policy.eval()
    policy.reset()
    for start in range(0, len(indices), batch_size):
        selected_indices = indices[start:start + batch_size]
        samples = [dataset[index] for index in selected_indices]
        processed = _preprocess_batch(samples, preprocessor, torch)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = postprocessor(policy.predict_action_chunk(processed))
        predicted = output.float().cpu().numpy()
        if predicted.shape != (len(samples), 10, 6) or not np.isfinite(predicted).all():
            raise RuntimeError(f"offline evaluationで非有限または不正shapeのaction: {start}")
        for local_index, sample in enumerate(samples):
            episode_index = int(sample["episode_index"])
            frame_index = int(sample["frame_index"])
            name = str(episode_index)
            labels = sample["action"].detach().cpu().numpy()
            pads = sample["action_is_pad"].detach().cpu().numpy().astype(bool)
            valid, _ = _validate_chunk_for_episode(
                teachers[episode_index]["actions"], labels, pads, frame_index,
            )
            if valid < 1:
                raise ValueError(f"episode {episode_index} frame {frame_index}に有効actionがありません")
            errors_by_episode[name].append(predicted[local_index, 0] - labels[0])
            chunk_errors_by_episode[name].append(predicted[local_index] - labels)
            pads_by_episode[name].append(pads)
    all_first_errors = np.concatenate([
        np.asarray(values, dtype=np.float64) for values in errors_by_episode.values()
    ])
    all_chunk_errors = np.concatenate([
        np.concatenate(values, axis=0)[~np.concatenate(pads_by_episode[name], axis=0)]
        for name, values in chunk_errors_by_episode.items()
    ])
    return {
        "frame_count": len(indices),
        "first_action": summarize_errors(all_first_errors),
        "whole_valid_chunk": summarize_errors(all_chunk_errors),
        "per_episode": {
            name: {
                "first_action": summarize_errors(np.asarray(errors_by_episode[name])),
                "whole_valid_chunk": summarize_errors(
                    np.concatenate(chunk_errors_by_episode[name], axis=0)[
                        ~np.concatenate(pads_by_episode[name], axis=0)
                    ]
                ),
            }
            for name in errors_by_episode
        },
    }


def _validate_chunk_for_episode(
    episode_actions: np.ndarray, labels: np.ndarray, pads: np.ndarray, frame_index: int,
) -> tuple[int, int]:
    valid_count = 0
    padded_count = 0
    for step, is_pad in enumerate(pads):
        target_index = frame_index + step
        if is_pad:
            if target_index < len(episode_actions):
                raise ValueError("episode内教師actionがpadding扱いです")
            padded_count += 1
        else:
            if target_index >= len(episode_actions) or not np.array_equal(
                labels[step], episode_actions[target_index],
            ):
                raise ValueError("training-studyの教師action chunkが元教師と不一致です")
            valid_count += 1
    return valid_count, padded_count


def _load_teacher_actions(study_path: Path, dataset_path: Path) -> list[dict]:
    collection_path = study_path.parent / "collection" / "collection.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if Path(collection["dataset"]).resolve() != dataset_path.resolve():
        raise ValueError("teacher collectionと学習datasetが一致しません")
    episodes = collection.get("episodes", [])
    if len(episodes) != 12:
        raise ValueError("元教師の12 episodeが必要です")
    result = []
    for index, episode in enumerate(episodes):
        source = Path(episode["source"])
        if source.name != f"train-{index:02d}-endpoint" or episode["frames"] != 450:
            raise ValueError("元教師episodeが固定splitと一致しません")
        replay, actions = load_endpoint_replay(source)
        result.append({
            "actions": actions,
            "initial_qpos": replay.qpos[0].copy(),
        })
    return result


def _baseline_metrics(path: Path, split: dict, baseline_training: dict) -> dict:
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "complete" or diagnostic.get("training_performed") is not False:
        raise ValueError("既存diagnosticが読み取り専用baselineではありません")
    if (
        Path(diagnostic["checkpoint"]["path"]).resolve()
        != Path(baseline_training["checkpoint"]).resolve()
        or Path(diagnostic["dataset"]["path"]).resolve()
        != Path(baseline_training["dataset"]).resolve()
        or diagnostic["dataset"]["fingerprint"] != baseline_training["dataset_fingerprint"]
        or diagnostic["inference"]["action_contract"] != ENDPOINT_MODE
    ):
        raise ValueError("既存diagnosticとbaseline学習記録のmodel/dataset/contractが一致しません")
    if diagnostic["checkpoint"]["files_sha256"] != checkpoint_fingerprint(
        Path(baseline_training["checkpoint"]),
    ):
        raise ValueError("既存baseline checkpointがdiagnostic実行後に変更されています")
    by_episode = diagnostic["action_fit"]["by_episode"]
    train_names = [
        f"train-{index:02d}-endpoint"
        for index in split["train_episode_indices"]
    ]
    validation_names = [
        f"train-{index:02d}-endpoint"
        for index in split["validation_episode_indices"]
    ]
    missing = (set(train_names) | set(validation_names)) - set(by_episode)
    if missing:
        raise ValueError(f"既存baselineにepisode指標がありません: {sorted(missing)}")
    return {
        "checkpoint_path": diagnostic["checkpoint"]["path"],
        "diagnostics_path": str(path.resolve()),
        "metrics_were_measured_on_full_dataset": True,
        "validation_was_seen_during_baseline_training": True,
        "training_episodes": {name: by_episode[name] for name in train_names},
        "validation_episodes": {name: by_episode[name] for name in validation_names},
        "note": "baselineは12 episodeすべてで学習済み。internal validation性能の独立baselineではなく、既存モデルの比較参考値。",
    }


def _preflight_initial_action(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    teacher_actions: list[dict],
) -> dict:
    import mujoco
    import torch

    start = CENTRAL_EPISODE * 450
    sample = dataset[start]
    if int(sample["episode_index"]) != CENTRAL_EPISODE or int(sample["frame_index"]) != 0:
        raise ValueError("中央位置preflightのdataset sampleが一致しません")
    if not np.allclose(
        sample["observation.state"].detach().cpu().numpy(),
        teacher_actions[CENTRAL_EPISODE]["initial_qpos"][:6],
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("中央位置preflightの初期関節状態が教師軌道と一致しません")
    model_path = Path(__file__).resolve().parents[2] / "assets" / "so101_dice_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    previous_action = sample["observation.state"].detach().cpu().numpy().astype(np.float64)
    if model.nu != 6:
        raise ValueError("preflight MuJoCoモデルのaction数が6ではありません")
    policy.eval()
    policy.reset()
    torch.manual_seed(SEED)
    processed = _preprocess_batch([sample], preprocessor, torch)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw = postprocessor(policy.select_action(processed)).squeeze(0).float().cpu().numpy()
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError("中央位置preflightの初期actionが非有限または不正です")
    applied = bounded_action(raw.astype(np.float64), previous_action, model.actuator_ctrlrange)
    if not np.isfinite(applied).all():
        raise ValueError("中央位置preflight後の安全補正actionが非有限です")
    lower = model.actuator_ctrlrange[:, 0].astype(np.float64)
    upper = model.actuator_ctrlrange[:, 1].astype(np.float64)
    lower[:5] += .1
    upper[:5] -= .1
    if np.any(applied < lower) or np.any(applied > upper):
        raise ValueError("中央位置preflight actionが安全関節範囲外です")
    return {
        "passed": True,
        "physical_rollout_executed": False,
        "episode_index": CENTRAL_EPISODE,
        "offset_mm": [0.0, 0.0],
        "raw_action_rad": raw.tolist(),
        "bounded_action_rad": applied.tolist(),
        "teacher_previous_action_rad": previous_action.tolist(),
        "max_action_delta_rad": float(np.max(np.abs(applied - previous_action))),
        "finite": True,
        "joint_margin_rad": .1,
        "per_cycle_delta_limit_rad": .05,
        "notice": "初期観測に対する非物理preflight。接触・長時間動作の安全性は既存評価器が確認する。",
    }


def run_training_study(
    directory: Path,
    study_path: Path,
    baseline_training_path: Path,
    baseline_diagnostics_path: Path,
    epochs: int = MAX_EPOCHS,
) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if epochs != MAX_EPOCHS:
        raise ValueError("今回の固定学習studyは最大3 epochに限定されます")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    baseline_training = json.loads(baseline_training_path.read_text(encoding="utf-8"))
    require_training_contract(baseline_training, ENDPOINT_MODE)
    dataset_path = Path(baseline_training["dataset"])
    if Path(study["dataset"]).resolve() != dataset_path.resolve():
        raise ValueError("studyとbaselineのdatasetが一致しません")
    dataset = LeRobotDataset(
        baseline_training["repo_id"],
        root=dataset_path,
        delta_timestamps={"action": [index / FPS for index in range(10)]},
    )
    current_dataset_fingerprint = dataset_fingerprint(dataset_path)
    if current_dataset_fingerprint != baseline_training.get("dataset_fingerprint"):
        raise ValueError("baseline学習記録と現在のdataset fingerprintが一致しません")
    split, train_indices, validation_indices = _resolve_episode_split(dataset, study)
    train_episode_indices = split["train_episode_indices"]
    validation_episode_indices = split["validation_episode_indices"]
    normalization_values = _training_only_stats(dataset_path, train_episode_indices)
    dataset_stats = training_normalization_stats(
        {"normalization_stats": normalization_values}, dataset,
    )
    stats_fingerprint = _fingerprint_json(normalization_values)
    teachers = _load_teacher_actions(study_path, dataset_path)
    baseline = _baseline_metrics(baseline_diagnostics_path, split, baseline_training)

    models = baseline_training["models"]
    torch.manual_seed(SEED)
    checkpoint = directory / "checkpoint"
    policy, preprocessor, postprocessor = load_policy(
        models, dataset, dataset_stats=dataset_stats,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=policy.config.optimizer_lr,
        betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    generator = np.random.default_rng(SEED)
    epoch_records = []
    best_validation_mae = float("inf")
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
            policy.train()
            losses = []
            seen = []
            batches = _sample_batches(train_indices, BATCH_SIZE, generator)
            for batch_index, indices in enumerate(batches):
                samples = [dataset[index] for index in indices]
                batch = preprocessor(torch.utils.data.default_collate(samples))
                if "action_is_pad" in batch:
                    batch["actions_id_pad"] = batch["action_is_pad"]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _ = policy.forward(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"epoch {epoch}で学習lossが非有限です")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 10., error_if_nonfinite=True,
                )
                optimizer.step()
                losses.append(float(loss.detach()))
                seen.extend(indices)
                if batch_index % 100 == 0 or batch_index + 1 == len(batches):
                    print(
                        f"training-study epoch {epoch}/{epochs}, "
                        f"update {batch_index + 1}/{len(batches)}, "
                        f"loss={losses[-1]:.6f}, grad={float(gradient_norm):.5f}",
                        flush=True,
                    )
            if len(seen) != 4500 or len(set(seen)) != 4500:
                raise RuntimeError("epoch内の学習データが全4500 sampleを一度ずつ使っていません")
            train_metrics = _evaluate_indices(
                policy, preprocessor, postprocessor, dataset, train_indices,
                teachers, BATCH_SIZE,
            )
            validation_metrics = _evaluate_indices(
                policy, preprocessor, postprocessor, dataset, validation_indices,
                teachers, BATCH_SIZE,
            )
            validation_mae = validation_metrics["first_action"]["mae_rad"]
            improved = validation_mae < best_validation_mae
            record = {
                "epoch": epoch,
                "updates": len(batches),
                "samples_seen": len(seen),
                "mean_training_loss": float(np.mean(losses)),
                "min_training_loss": float(np.min(losses)),
                "train_action_fit": train_metrics,
                "validation_action_fit": validation_metrics,
                "selected_as_best": improved,
            }
            epoch_records.append(record)
            with (directory / "epochs.csv").open(
                "x" if epoch == 1 else "a", encoding="utf-8", newline="",
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "epoch", "updates", "samples_seen", "mean_training_loss",
                    "min_training_loss", "train_first_action_mae_rad",
                    "train_first_action_rmse_rad", "validation_first_action_mae_rad",
                    "validation_first_action_rmse_rad", "validation_first_action_max_abs_rad",
                    "selected_as_best",
                ])
                if epoch == 1:
                    writer.writeheader()
                writer.writerow({
                    "epoch": epoch,
                    "updates": len(batches),
                    "samples_seen": len(seen),
                    "mean_training_loss": record["mean_training_loss"],
                    "min_training_loss": record["min_training_loss"],
                    "train_first_action_mae_rad": train_metrics["first_action"]["mae_rad"],
                    "train_first_action_rmse_rad": train_metrics["first_action"]["rmse_rad"],
                    "validation_first_action_mae_rad": validation_mae,
                    "validation_first_action_rmse_rad": validation_metrics["first_action"]["rmse_rad"],
                    "validation_first_action_max_abs_rad": validation_metrics["first_action"]["max_abs_rad"],
                    "selected_as_best": improved,
                })
            if improved:
                policy.save_pretrained(checkpoint)
                best_validation_mae = validation_mae
                best_epoch = epoch
                print(
                    f"best validation checkpoint: epoch={epoch}, "
                    f"MAE={validation_mae:.6f} rad",
                    flush=True,
                )

    if best_epoch is None:
        raise RuntimeError("validation checkpointが選択されませんでした")
    contract_path = checkpoint / "action_contract.json"
    contract_path.write_text(
        json.dumps(baseline_training["action_contract"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    policy, preprocessor, postprocessor = load_policy(
        models, dataset, checkpoint, dataset_stats=dataset_stats,
    )
    preflight = _preflight_initial_action(policy, preprocessor, postprocessor, dataset, teachers)
    manifest = {
        **split,
        "dataset_path": str(dataset_path.resolve()),
        "study_path": str(study_path.resolve()),
        "study_sha256": _sha256(study_path),
        "dataset_fingerprint": current_dataset_fingerprint,
        "normalization_stats": normalization_values,
        "normalization_stats_sha256": stats_fingerprint,
    }
    write_json(directory / "split.json", manifest)
    write_json(directory / "preflight.json", preflight)
    checkpoint_files = checkpoint_fingerprint(checkpoint)
    training = {
        "study_type": "fixed_internal_validation_action_fit",
        "status": "complete",
        "dataset": str(dataset_path.resolve()),
        "repo_id": baseline_training["repo_id"],
        "models": models,
        "checkpoint": str(checkpoint.resolve()),
        "uploaded": False,
        "seed": SEED,
        "action_contract": baseline_training["action_contract"],
        "dataset_fingerprint": current_dataset_fingerprint,
        "normalization_stats": normalization_values,
        "normalization_stats_sha256": stats_fingerprint,
        "normalization_stats_source_episode_indices": train_episode_indices,
        "official_held_out_positions_used": False,
        "epochs_requested": epochs,
        "epochs_completed": len(epoch_records),
        "batch_size": BATCH_SIZE,
        "optimizer_state_restored": False,
        "optimizer_state_continuous_across_epochs": True,
        "training_episode_indices": train_episode_indices,
        "validation_episode_indices": validation_episode_indices,
        "central_physics_evaluation_episode": CENTRAL_EPISODE,
        "best_epoch": best_epoch,
        "best_validation_first_action_mae_rad": best_validation_mae,
        "baseline": baseline,
        "epochs": epoch_records,
        "preflight": preflight,
        "elapsed_s": time.perf_counter() - started,
        "checkpoint_files_sha256": checkpoint_files,
        "checkpoint_sha256": _fingerprint_json(checkpoint_files),
        "notice": "新規base SmolVLAから既存10教師episodeだけで学習。validation 2 episodeはoptimizer/normalization統計に不使用。",
    }
    write_json(directory / "training.json", training)
    return training


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="固定internal splitでSmolVLAを学習しepochごとのaction fitを評価",
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline-training", type=Path, required=True)
    parser.add_argument("--baseline-diagnostics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/training-studies"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        training = run_training_study(
            directory,
            args.study,
            args.baseline_training,
            args.baseline_diagnostics,
            args.epochs,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        write_json(directory / "error.json", {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        })
        print(f"学習位置内validation studyエラー: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"学習study保存先: {directory.resolve()}", flush=True)
    print(
        f"best epoch={training['best_epoch']}, "
        f"validation MAE={training['best_validation_first_action_mae_rad']:.6g} rad",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
