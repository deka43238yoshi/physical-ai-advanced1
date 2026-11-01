import json

import numpy as np
import pytest

from so101_dice_sim import training_position_evaluation as evaluation
from so101_dice_sim.learning_study import split_positions
from so101_dice_sim.visualization import Replay


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    teacher_path = study_dir / "train-04-endpoint"
    teacher_path.mkdir()
    expected_qpos = np.array([0, 0, 0, 0, 0, .6, .245, .015, .016, 1, 0, 0, 0.])
    report = {
        "requested_face": 2, "initial_qpos": expected_qpos.tolist(),
        "mode": "endpoint_linear_bounded",
    }
    teacher = Replay(report, [{"top_face": 1}], np.array([0.]), expected_qpos[None, :], {})
    (teacher_path / "replay.json").write_text("teacher", encoding="utf-8")
    (teacher_path / "trajectory.csv").write_text("teacher", encoding="utf-8")
    study = {
        "split": split_positions(),
        "results": [
            {
                "group": group, "index": index, "offset_mm": offset, "passed": True,
                "source": str((study_dir / f"{group}-{index:02d}-endpoint").resolve()),
            }
            for group, positions in (
                ("train", split_positions()["train_offsets_mm"]),
                ("evaluation", split_positions()["evaluation_offsets_mm"]),
            )
            for index, offset in enumerate(positions)
        ],
        "dataset": str((tmp_path / "dataset").resolve()),
    }
    study_path = study_dir / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    training = {
        "dataset": study["dataset"], "repo_id": "local/test",
        "checkpoint": str(checkpoint),
    }
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    monkeypatch.setattr(evaluation, "require_training_contract", lambda *args: {})
    monkeypatch.setattr(evaluation, "load_endpoint_replay", lambda path: (teacher, None))
    return study_path, training_path, teacher_path, teacher


def test_only_verified_central_training_position_is_selected(inputs):
    study_path, training_path, teacher_path, teacher = inputs
    study, _, selected_path, selected_teacher, offset = evaluation.load_training_position(
        study_path, training_path,
    )
    assert selected_path == teacher_path
    assert selected_teacher is teacher
    assert offset == [0, 0]
    assert study["split"]["train_offsets_mm"][evaluation.TRAINING_POSITION_INDEX] == [0, 0]


@pytest.mark.parametrize("change", ["position", "group", "index", "passed", "source"])
def test_rejects_changed_or_unverified_teacher_selection(inputs, change):
    study_path, training_path, _, _ = inputs
    study = json.loads(study_path.read_text(encoding="utf-8"))
    row = study["results"][evaluation.TRAINING_POSITION_INDEX]
    if change == "position":
        row["offset_mm"] = [2.5, 2.5]
    elif change == "group":
        row["group"] = "evaluation"
    elif change == "index":
        row["index"] = 3
    elif change == "passed":
        row["passed"] = False
    else:
        row["source"] = str(study_path.parent / "other")
    study_path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_training_position(study_path, training_path)


def test_rejects_changed_split_and_training_dataset(inputs, monkeypatch):
    study_path, training_path, _, _ = inputs
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["split"]["train_offsets_mm"][4] = [1, 1]
    study_path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_training_position(study_path, training_path)

    study["split"] = split_positions()
    study_path.write_text(json.dumps(study), encoding="utf-8")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["dataset"] = str(study_path.parent / "other-dataset")
    training_path.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="学習データ"):
        evaluation.load_training_position(study_path, training_path)


def test_rejects_wrong_face_or_initial_face(inputs, monkeypatch):
    study_path, training_path, _, teacher = inputs
    teacher.report["requested_face"] = 1
    with pytest.raises(ValueError, match="初期上面1・目標面2"):
        evaluation.load_training_position(study_path, training_path)
    teacher.report["requested_face"] = 2
    teacher.rows[0]["top_face"] = 2
    with pytest.raises(ValueError, match="初期上面1・目標面2"):
        evaluation.load_training_position(study_path, training_path)


def test_rejects_dataset_contract_failure(inputs, monkeypatch):
    study_path, training_path, _, _ = inputs
    monkeypatch.setattr(
        evaluation, "require_training_contract",
        lambda *args: (_ for _ in ()).throw(ValueError("wrong action contract")),
    )
    with pytest.raises(ValueError, match="wrong action contract"):
        evaluation.load_training_position(study_path, training_path)
