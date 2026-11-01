import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .action_contract import ENDPOINT_MODE, require_training_contract
from .endpoint_collection import load_endpoint_replay
from .learning import load_policy, training_normalization_stats
from .learning_study import split_positions
from .policy_evaluation import evaluate_position
from .reporting import create_run_directory, write_json


TRAINING_POSITION_INDEX = 4


def load_training_position(study_path: Path, training_path: Path):
    study = json.loads(study_path.read_text(encoding="utf-8"))
    expected_split = json.loads(json.dumps(split_positions()))
    if study["split"] != expected_split:
        raise ValueError("学習位置分割が事前固定した分割と一致しません")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    require_training_contract(training, ENDPOINT_MODE)
    if Path(training["dataset"]).resolve() != Path(study["dataset"]).resolve():
        raise ValueError("評価するモデルが指定した学習データに対応しません")
    offset = expected_split["train_offsets_mm"][TRAINING_POSITION_INDEX]
    expected_results = [
        (group, index, position)
        for group, positions in (
            ("train", expected_split["train_offsets_mm"]),
            ("evaluation", expected_split["evaluation_offsets_mm"]),
        )
        for index, position in enumerate(positions)
    ]
    results = {
        (row["group"], row["index"]): row for row in study["results"]
    }
    if len(study["results"]) != len(expected_results) or len(results) != len(expected_results):
        raise ValueError("学習・評価位置の教師検証数が一致しません")
    for group, index, position in expected_results:
        row = results.get((group, index))
        if row is None or row["offset_mm"] != position or row["passed"] is not True:
            raise ValueError("位置分割に対応する成功教師検証がありません")
    teacher_result = results[("train", TRAINING_POSITION_INDEX)]
    if (
        teacher_result["group"] != "train"
        or teacher_result["index"] != TRAINING_POSITION_INDEX
        or teacher_result["offset_mm"] != offset
        or teacher_result["passed"] is not True
        or teacher_result["source"] != str(
            (study_path.parent / f"train-{TRAINING_POSITION_INDEX:02d}-endpoint").resolve()
        )
    ):
        raise ValueError("中央学習位置の成功教師が事前検証結果と一致しません")
    teacher_path = Path(teacher_result["source"])
    teacher, _ = load_endpoint_replay(teacher_path)
    if (
        teacher.report["requested_face"] != 2
        or not np.array_equal(teacher.qpos[0], np.asarray(teacher.report["initial_qpos"]))
        or int(teacher.rows[0]["top_face"]) != 1
    ):
        raise ValueError("教師は初期上面1・目標面2の成功軌道ではありません")
    return study, training, teacher_path, teacher, offset


def evaluate_training_position(directory: Path, study_path: Path, training_path: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    study, training, teacher_path, teacher, offset = load_training_position(
        study_path, training_path,
    )
    dataset = LeRobotDataset(training["repo_id"], root=Path(training["dataset"]))
    policy, pre, post = load_policy(
        training["models"], dataset, Path(training["checkpoint"]),
        dataset_stats=training_normalization_stats(training, dataset),
    )
    write_json(directory / "settings.json", {
        "evaluation_type": "training_position_diagnostic",
        "target_face": 2,
        "training_position_index": TRAINING_POSITION_INDEX,
        "evaluation_offset_mm": offset,
        "teacher": str(teacher_path.resolve()),
        "study": str(study_path.resolve()),
        "training": str(training_path.resolve()),
        "dataset": str(Path(training["dataset"]).resolve()),
        "checkpoint": str(Path(training["checkpoint"]).resolve()),
        "seed": 42,
        "horizon_s": 45,
        "split": study["split"],
        "study_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "teacher_sha256": {
            name: hashlib.sha256((teacher_path / name).read_bytes()).hexdigest()
            for name in ("replay.json", "trajectory.csv")
        },
        "initial_qpos": teacher.qpos[0].tolist(),
    })
    evaluation_dir = directory / "evaluation-00"
    evaluation_dir.mkdir()
    report = evaluate_position(
        evaluation_dir, policy, pre, post, offset, 42,
        expected_initial_qpos=teacher.qpos[0],
        evaluation_type="training_position_diagnostic",
    )
    summary = {
        "runs": 1, "passed_count": int(report["passed"]),
        "results": [report], "uploaded": False,
        "notice": "学習位置(0,0)mmの診断。教師は比較基準だけに使用し、方策入力・動作には使用しない。",
    }
    write_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="面2SmolVLAを中央学習位置1件で教師と同じ初期状態から診断評価",
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/training-position-evaluations"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        result = evaluate_training_position(directory, args.study, args.training)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        write_json(directory / "error.json", {
            "status": "error", "error_type": type(error).__name__, "message": str(error),
        })
        print(f"学習位置評価エラー: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"学習位置評価保存先: {directory.resolve()}", flush=True)
    return 0 if result["passed_count"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
