import json

import pytest

from so101_dice_sim.action_contract import (
    CONTRACT_NAME, ENDPOINT_MODE, LEGACY_MODE, action_contract,
    dataset_contract, dataset_fingerprint, require_training_contract,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_legacy_without_sidecar_remains_explicitly_supported(tmp_path):
    assert dataset_contract(tmp_path, LEGACY_MODE) == action_contract(LEGACY_MODE)
    assert require_training_contract({"dataset": str(tmp_path)}, LEGACY_MODE) == action_contract(LEGACY_MODE)
    with pytest.raises(ValueError):
        dataset_contract(tmp_path, ENDPOINT_MODE)


@pytest.mark.parametrize("mode", [LEGACY_MODE, ENDPOINT_MODE])
def test_mode_mismatch_is_rejected(tmp_path, mode):
    save(tmp_path / CONTRACT_NAME, action_contract(mode))
    assert dataset_contract(tmp_path, mode) == action_contract(mode)
    other = ENDPOINT_MODE if mode == LEGACY_MODE else LEGACY_MODE
    with pytest.raises(ValueError):
        dataset_contract(tmp_path, other)


def test_unknown_or_modified_contract_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        action_contract("unknown")
    contract = action_contract(ENDPOINT_MODE)
    contract["target_offset_s"] = .002
    save(tmp_path / CONTRACT_NAME, contract)
    with pytest.raises(ValueError):
        dataset_contract(tmp_path, ENDPOINT_MODE)


def test_checkpoint_contract_stats_and_data_are_bound(tmp_path):
    root = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    contract = action_contract(ENDPOINT_MODE)
    save(root / CONTRACT_NAME, contract)
    for path in ("meta/info.json", "meta/stats.json", "data/chunk/file.parquet"):
        save(root.joinpath(*path.split("/")), {"test": 1})
    save(checkpoint / CONTRACT_NAME, contract)
    training = {
        "dataset": str(root), "checkpoint": str(checkpoint),
        "action_contract": contract, "dataset_fingerprint": dataset_fingerprint(root),
    }
    assert require_training_contract(training, ENDPOINT_MODE) == contract
    with pytest.raises(ValueError):
        require_training_contract(training, LEGACY_MODE)
    save(checkpoint / CONTRACT_NAME, action_contract(LEGACY_MODE))
    with pytest.raises(ValueError):
        require_training_contract(training, ENDPOINT_MODE)
    save(checkpoint / CONTRACT_NAME, contract)
    save(root / "meta" / "stats.json", {"changed": 2})
    with pytest.raises(ValueError):
        require_training_contract(training, ENDPOINT_MODE)


def test_old_checkpoint_cannot_be_used_as_endpoint(tmp_path):
    save(tmp_path / CONTRACT_NAME, action_contract(ENDPOINT_MODE))
    with pytest.raises(ValueError):
        require_training_contract({"dataset": str(tmp_path)}, ENDPOINT_MODE)


def test_missing_dataset_is_not_fingerprinted_as_empty_success(tmp_path):
    with pytest.raises(ValueError):
        dataset_fingerprint(tmp_path)
