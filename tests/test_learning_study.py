import numpy as np
import pytest

from so101_dice_sim.learning import training_batches, main
from so101_dice_sim.learning_study import split_positions
from so101_dice_sim.policy_evaluation import evaluation_phase, require_initial_qpos
from so101_dice_sim.placement import stable_support_window


def test_split_is_fixed_unique_disjoint_and_within_existing_range():
    split = split_positions()
    train = {tuple(p) for p in split["train_offsets_mm"]}
    evaluation = {tuple(p) for p in split["evaluation_offsets_mm"]}
    assert len(train) == 12 and len(evaluation) == 4
    assert not train & evaluation
    assert all(abs(v) <= 5 for point in train | evaluation for v in point)
    assert split == split_positions()
    assert split["target_face"] == 2


@pytest.mark.parametrize("step,phase", [
    (0, "initial"), (8499, "policy"), (8500, "hold"), (9500, "hold"), (9501, "policy"),
    (11500, "rotation_hold"), (12500, "rotation_hold"), (12501, "policy"),
    (21999, "policy"), (22000, "final_evaluation"), (22500, "final_evaluation"),
])
def test_evaluation_windows_match_existing_sample_counts(step, phase):
    assert evaluation_phase(step) == phase


def test_current_support_window_not_stale_support_is_required():
    sample = {
        "dice_table_contact": True, "dice_table_force_n": .2,
        "dice_bottom_gap_m": 0, "dice_linear_speed_m_s": 0,
        "dice_angular_speed_rad_s": 0, "top_face": 2, "tilt_deg": 0,
    }
    window = [sample.copy() for _ in range(251)]
    assert stable_support_window(window)
    assert not stable_support_window(window[:-1])
    window[-1]["dice_table_contact"] = False
    assert not stable_support_window(window)


def test_training_position_initial_state_must_match_teacher_exactly():
    initial = np.array([0, 0, 0, 0, 0, .6, .245, .015, .016, 1, 0, 0, 0.])
    require_initial_qpos(initial, initial.copy())
    changed = initial.copy()
    changed[1] = 1e-12
    with pytest.raises(ValueError, match="完全一致"):
        require_initial_qpos(initial, changed)
    with pytest.raises(ValueError, match="完全一致"):
        require_initial_qpos(initial[:-1], initial)
    changed = initial.copy()
    changed[0] = np.nan
    with pytest.raises(ValueError, match="完全一致"):
        require_initial_qpos(changed, initial)


def test_window_counts():
    phases = [evaluation_phase(i) for i in range(22501)]
    assert phases.count("hold") == 1001
    assert phases.count("rotation_hold") == 1001
    assert phases.count("final_evaluation") == 501


def test_full_epoch_covers_every_sample_once_in_seeded_shuffled_order():
    batches = training_batches(5400, 3, 1, 4)
    assert len(batches) == 1350
    flattened = [i for batch in batches for i in batch]
    assert sorted(flattened) == list(range(5400))
    assert flattened != list(range(5400))
    assert batches == training_batches(5400, 3, 1, 4)


def test_epochs_preserve_last_partial_batch_and_independent_shuffle():
    batches = training_batches(9, 3, 2, 4)
    assert [len(batch) for batch in batches] == [4, 4, 1, 4, 4, 1]
    first = [i for batch in batches[:3] for i in batch]
    second = [i for batch in batches[3:] for i in batch]
    assert sorted(first) == sorted(second) == list(range(9))
    assert first != second
    assert training_batches(1350, 3, None, 1) == [[25], [475], [925]]


@pytest.mark.parametrize("args", [
    ["train", "--epochs", "1", "--steps", "3"],
    ["rollout", "--epochs", "1"],
    ["train", "--batch-size", "4"],
])
def test_full_training_cli_rejects_ambiguous_options(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_full_training_cli_forwards_epoch_and_batch(tmp_path, monkeypatch):
    seen = []

    def train(directory, dataset, models, steps, mode, *, epochs, batch_size):
        seen.append((epochs, batch_size, mode))

    monkeypatch.setattr("so101_dice_sim.learning.train", train)
    assert main([
        "train", "--dataset", str(tmp_path / "dataset"), "--models", str(tmp_path / "models.json"),
        "--epochs", "1", "--batch-size", "4", "--action-mode", "endpoint_linear_v2",
        "--output-dir", str(tmp_path),
    ]) == 0
    assert seen == [(1, 4, "endpoint_linear_v2")]
