import json
import hashlib
import ssl

import numpy as np
import pytest

from so101_dice_sim.learning import (
    bounded_action, build_ca_bundle, checkpoint_fingerprint, main, resolve_resume_training,
    teacher_indices,
)


def test_teacher_sampling_and_causal_target():
    indices = teacher_indices(22501, .002)
    assert len(indices) == 450 and indices[0] == 0 and indices[-1] == 22450
    assert all(index + 1 < 22501 for index in indices)


@pytest.mark.parametrize("count,step", [(1, .002), (10, 0), (10, float("nan")), (10, .003)])
def test_invalid_teacher_timing(count, step):
    with pytest.raises(ValueError):
        teacher_indices(count, step)


def test_action_bounds_and_rate_limit_are_explicit():
    ranges = np.array([[-1., 1.]] * 6)
    previous = np.zeros(6)
    output = bounded_action(np.array([2., -2., .02, 0, 0, .6]), previous, ranges)
    np.testing.assert_allclose(output, [.05, -.05, .02, 0, 0, .05])
    np.testing.assert_array_equal(previous, np.zeros(6))


def test_nonfinite_action_rejected():
    with pytest.raises(ValueError):
        bounded_action(np.full(6, np.nan), np.zeros(6), np.array([[-1, 1]] * 6))


def test_unsafe_previous_or_reversed_limits_rejected():
    with pytest.raises(ValueError):
        bounded_action(np.zeros(6), np.full(6, 2.), np.array([[-1, 1]] * 6))
    with pytest.raises(ValueError):
        bounded_action(np.zeros(6), np.zeros(6), np.array([[1., -1.]] * 6))


@pytest.mark.parametrize("stage", ["collect", "collect-endpoints", "train", "rollout"])
def test_missing_stage_arguments(stage):
    with pytest.raises(SystemExit) as error:
        main([stage])
    assert error.value.code == 2


@pytest.mark.parametrize("pem", [False, True])
def test_ca_bundle_keeps_default_trust_and_records_fingerprint(tmp_path, pem):
    der = ssl.create_default_context().get_ca_certs(binary_form=True)[0]
    certificate = ssl.DER_cert_to_PEM_cert(der)
    default = tmp_path / "default.pem"
    default.write_text(certificate, encoding="ascii")
    extra = tmp_path / "extra.cer"
    extra.write_bytes(certificate.encode("ascii") if pem else der)
    output = tmp_path / "bundle.pem"
    assert build_ca_bundle(default, extra, output) == hashlib.sha256(der).hexdigest()
    assert output.read_text(encoding="ascii").count("-----BEGIN CERTIFICATE-----") == 2
    ssl.create_default_context(cafile=str(output))


def test_invalid_ca_rejected(tmp_path):
    default = tmp_path / "default.pem"
    default.write_text("", encoding="ascii")
    extra = tmp_path / "invalid.cer"
    extra.write_bytes(b"not a certificate")
    with pytest.raises(ssl.SSLError):
        build_ca_bundle(default, extra, tmp_path / "bundle.pem")


def test_ca_option_not_silently_ignored():
    with pytest.raises(SystemExit) as error:
        main(["gpu", "--ca-cert", "unused.cer"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["next_step_hold_v1", "endpoint_linear_v2"])
def test_training_cli_forwards_explicit_action_contract(tmp_path, monkeypatch, mode):
    seen = []

    def train(directory, dataset, models, steps, action_mode):
        seen.append((dataset, models, steps, action_mode))

    monkeypatch.setattr("so101_dice_sim.learning.train", train)
    assert main([
        "train", "--dataset", str(tmp_path / "dataset"), "--models", str(tmp_path / "models.json"),
        "--action-mode", mode, "--output-dir", str(tmp_path),
    ]) == 0
    assert seen == [(tmp_path / "dataset", tmp_path / "models.json", 3, mode)]


def test_endpoint_collection_mode_cannot_be_silently_overridden():
    with pytest.raises(SystemExit) as error:
        main(["collect", "--sources", "old", "--action-mode", "endpoint_linear_v2"])
    assert error.value.code == 2


def test_training_cli_forwards_single_resume_epoch(tmp_path, monkeypatch):
    seen = []

    def train(directory, dataset, models, steps, mode, *, epochs, batch_size, resume_training_path):
        seen.append((dataset, models, mode, epochs, batch_size, resume_training_path))

    monkeypatch.setattr("so101_dice_sim.learning.train", train)
    dataset = tmp_path / "dataset"
    resume = tmp_path / "training.json"
    assert main([
        "train", "--dataset", str(dataset), "--resume-training", str(resume),
        "--epochs", "1", "--batch-size", "4", "--action-mode", "endpoint_linear_v2",
        "--output-dir", str(tmp_path),
    ]) == 0
    assert seen == [(dataset, None, "endpoint_linear_v2", 1, 4, resume)]


@pytest.mark.parametrize("extra", [
    ["--epochs", "2", "--batch-size", "4", "--action-mode", "endpoint_linear_v2"],
    ["--epochs", "1", "--batch-size", "1", "--action-mode", "endpoint_linear_v2"],
    ["--epochs", "1", "--batch-size", "4", "--action-mode", "next_step_hold_v1"],
])
def test_resume_cli_only_accepts_one_endpoint_epoch_at_batch_four(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        main([
            "train", "--dataset", str(tmp_path / "dataset"),
            "--resume-training", str(tmp_path / "training.json"), *extra,
        ])
    assert error.value.code == 2


def test_resume_checkpoint_must_match_action_contract_dataset_and_base(
    tmp_path, monkeypatch,
):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    dataset.mkdir()
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    models = {"policy": {"path": "base"}, "backbone": {"path": "vlm"}}
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(models), encoding="utf-8")
    record = {
        "epochs": 1, "steps": 3, "samples_seen": 9,
        "dataset": str(dataset.resolve()), "checkpoint": str(checkpoint.resolve()),
        "models": models, "monitored_weights_changed": True,
    }
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr("so101_dice_sim.learning.require_training_contract", lambda *args: {})
    previous, actual_checkpoint, fingerprints = resolve_resume_training(
        training_path, dataset, models_path, "endpoint_linear_v2", 1,
    )
    assert previous["epochs"] == 1 and actual_checkpoint == checkpoint
    assert fingerprints == checkpoint_fingerprint(checkpoint)
    with pytest.raises(ValueError, match="1epoch"):
        resolve_resume_training(training_path, dataset, models_path, "endpoint_linear_v2", 2)
    record["initialized_from_training"] = str(training_path)
    training_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="同じデータセット"):
        resolve_resume_training(training_path, dataset, models_path, "endpoint_linear_v2", 1)
    record.pop("initialized_from_training")
    training_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="同じデータセット"):
        resolve_resume_training(
            training_path, tmp_path / "other", models_path, "endpoint_linear_v2", 1,
        )
    models_path.write_text(json.dumps({"different": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="base model"):
        resolve_resume_training(training_path, dataset, models_path, "endpoint_linear_v2", 1)
