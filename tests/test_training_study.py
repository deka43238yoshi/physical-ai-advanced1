import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_dice_sim.learning import checkpoint_fingerprint, training_normalization_stats
from so101_dice_sim.learning_study import split_positions
from so101_dice_sim.training_study import (
    BATCH_SIZE,
    VALIDATION_EPISODES,
    _baseline_metrics,
    _resolve_episode_split,
    _sample_batches,
    _validate_chunk_for_episode,
)


def _fake_dataset():
    episodes = [
        {
            "episode_index": index,
            "dataset_from_index": index * 450,
            "dataset_to_index": (index + 1) * 450,
            "length": 450,
        }
        for index in range(12)
    ]
    return SimpleNamespace(
        meta=SimpleNamespace(episodes=episodes),
        num_episodes=12,
        num_frames=5400,
    )


def test_fixed_split_keeps_internal_holdout_and_central_episode_in_training():
    study = {"split": json.loads(json.dumps(split_positions()))}

    manifest, train, validation = _resolve_episode_split(_fake_dataset(), study)

    assert manifest["validation_episode_indices"] == list(VALIDATION_EPISODES)
    assert manifest["central_physics_evaluation_episode"] == 4
    assert 4 in manifest["train_episode_indices"]
    assert manifest["official_held_out_positions_used"] is False
    assert len(train) == 4500
    assert len(validation) == 900
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(5400))


def test_fixed_split_rejects_reordered_or_noncontiguous_episode_ranges():
    dataset = _fake_dataset()
    dataset.meta.episodes[2]["dataset_from_index"] = 901

    with pytest.raises(ValueError, match="並びまたはframe範囲"):
        _resolve_episode_split(dataset, {"split": json.loads(json.dumps(split_positions()))})


def test_each_epoch_batch_order_is_seeded_and_covers_each_training_sample_once():
    indices = list(range(4500))

    first = _sample_batches(indices, BATCH_SIZE, np.random.default_rng(42))
    repeated = _sample_batches(indices, BATCH_SIZE, np.random.default_rng(42))

    assert first == repeated
    assert len(first) == 1125
    assert all(len(batch) == BATCH_SIZE for batch in first)
    assert sorted(index for batch in first for index in batch) == indices


def test_action_chunk_is_verified_against_teacher_targets_and_padding():
    actions = np.arange(18, dtype=np.float32).reshape(3, 6)
    labels = np.vstack((actions[1], actions[2], actions[2]))
    pads = np.array([False, False, True])

    assert _validate_chunk_for_episode(actions, labels, pads, 1) == (2, 1)

    labels[0, 0] += 1
    with pytest.raises(ValueError, match="教師action"):
        _validate_chunk_for_episode(actions, labels, pads, 1)


def test_train_only_stats_are_validated_as_state_and_action_arrays():
    recorded = {
        "normalization_stats": {
            name: {"mean": [0.1] * 6, "std": [0.2] * 6, "count": 4500}
            for name in ("observation.state", "action")
        },
    }

    stats = training_normalization_stats(recorded, dataset=object())

    assert set(stats) == {"observation.state", "action"}
    np.testing.assert_array_equal(stats["action"]["mean"], np.full(6, .1, np.float32))
    np.testing.assert_array_equal(stats["observation.state"]["std"], np.full(6, .2, np.float32))

    recorded["normalization_stats"]["action"]["std"][0] = 0
    with pytest.raises(ValueError, match="正規化統計が不正"):
        training_normalization_stats(recorded, dataset=object())


def test_baseline_metrics_require_same_checkpoint_dataset_and_diagnostic(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"test weights")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    fingerprint = {"meta/info.json": "dataset-hash"}
    split = {
        "train_episode_indices": [0, 2],
        "validation_episode_indices": [1],
    }
    episodes = {
        f"train-{index:02d}-endpoint": {"first_action": {"mae_rad": .1}}
        for index in range(3)
    }
    baseline_training = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "dataset_fingerprint": fingerprint,
    }
    diagnostic_path = tmp_path / "diagnostics.json"
    diagnostic_path.write_text(json.dumps({
        "status": "complete",
        "training_performed": False,
        "checkpoint": {
            "path": str(checkpoint),
            "files_sha256": checkpoint_fingerprint(checkpoint),
        },
        "dataset": {"path": str(dataset), "fingerprint": fingerprint},
        "inference": {"action_contract": "endpoint_linear_v2"},
        "action_fit": {"by_episode": episodes},
    }), encoding="utf-8")

    metrics = _baseline_metrics(diagnostic_path, split, baseline_training)

    assert metrics["validation_was_seen_during_baseline_training"] is True
    assert set(metrics["training_episodes"]) == {
        "train-00-endpoint", "train-02-endpoint",
    }
    assert set(metrics["validation_episodes"]) == {"train-01-endpoint"}

    baseline_training["dataset_fingerprint"] = {"changed": "fingerprint"}
    with pytest.raises(ValueError, match="model/dataset/contract"):
        _baseline_metrics(diagnostic_path, split, baseline_training)
