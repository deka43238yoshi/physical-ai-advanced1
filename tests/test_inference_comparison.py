import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from so101_dice_sim import inference_comparison
from so101_dice_sim.inference_comparison import (
    DRAWS, EPISODES, FRAMES, action_metrics, control_indices, forbid_physics_steps,
    mean_four, noise_seed, paired_effect, require_initial_match,
    require_reference_coverage, sampling_steps,
)


def record(source="teacher", frame=0, draw=0, action=None):
    return {
        "source": source, "episode": 4, "frame": frame, "draw": draw,
        "seeds": [noise_seed(4, frame, draw)], "noise_sha256": ["shared"], "steps": 10,
        "action_rad": np.zeros(6) if action is None else np.asarray(action),
        "target_rad": np.zeros(6), "bounded_estimate_rad": np.zeros(6),
        "policy_call_ms": 2.0,
    }


def test_fixed_samples_exclude_official_hold_out_and_use_new_shared_seeds():
    assert EPISODES == (4, 1, 11)
    assert len(FRAMES) == 40
    assert FRAMES[:31] == tuple(range(31))
    assert FRAMES[-1] == 449
    seeds = [noise_seed(episode, frame, draw)
             for episode in EPISODES for frame in FRAMES for draw in range(DRAWS)]
    assert len(set(seeds)) == 480
    assert min(seeds) > 105
    assert noise_seed(4, 0, 0) == 7306
    assert noise_seed(4, 0, 1) == 7307


@pytest.mark.parametrize("arguments", [
    (0, 0, 0), (4, 31, 0), (4, 0, 4), (True, 0, 0), (4, 0.0, 0),
])
def test_noise_selection_rejects_outside_fixed_protocol(arguments):
    with pytest.raises(ValueError, match="事前固定"):
        noise_seed(*arguments)


def test_mean_four_averages_raw_actions_without_selecting_or_clipping():
    actions = np.array([[.4] * 6, [.8] * 6, [-.1] * 6, [-.3] * 6])
    np.testing.assert_allclose(mean_four(actions), [.2] * 6)
    np.testing.assert_array_equal(actions[0], [.4] * 6)
    for invalid in (np.zeros((3, 6)), np.full((4, 6), np.nan)):
        with pytest.raises(ValueError, match="有限な4x6"):
            mean_four(invalid)


def test_sampler_configuration_restored_after_success_and_error():
    config = SimpleNamespace(num_steps=10)
    policy = SimpleNamespace(config=config, model=SimpleNamespace(config=config))
    with sampling_steps(policy, 50):
        assert policy.config.num_steps == 50
    assert config.num_steps == 10
    with pytest.raises(RuntimeError, match="inference error"):
        with sampling_steps(policy, 50):
            raise RuntimeError("inference error")
    assert config.num_steps == 10
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="正の整数"):
            with sampling_steps(policy, invalid):
                pass
    assert config.num_steps == 10


def test_step_guard_rejects_all_physics_entrypoints_and_restores_them():
    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2")}
    with forbid_physics_steps():
        for name in originals:
            with pytest.raises(RuntimeError, match="物理stepは禁止"):
                getattr(mujoco, name)(None, None)
    assert all(getattr(mujoco, name) is original for name, original in originals.items())
    with pytest.raises(RuntimeError, match="物理stepは禁止"):
        with forbid_physics_steps():
            mujoco.mj_step(None, None)
    assert all(getattr(mujoco, name) is original for name, original in originals.items())


def test_initial_action_calibration_rejects_nonmatching_or_invalid_values():
    expected = np.arange(6, dtype=float)
    assert require_initial_match(expected, expected) == 0
    with pytest.raises(ValueError, match="再現できません"):
        require_initial_match(expected + .001, expected)
    for invalid in (np.zeros(5), np.full(6, np.nan)):
        with pytest.raises(ValueError, match="有限な6関節"):
            require_initial_match(invalid, expected)


def test_reference_must_have_full_dataset_coverage_and_original_statistics():
    training = {"samples_seen": 5400, "losses": [{"sample_indices": list(range(5400))}]}
    require_reference_coverage(training)
    training["losses"][0]["sample_indices"][-1] = 0
    with pytest.raises(ValueError, match="全5,400 frame"):
        require_reference_coverage(training)
    training["losses"][0]["sample_indices"][-1] = 5399
    training["normalization_stats"] = {}
    with pytest.raises(ValueError, match="全dataset統計"):
        require_reference_coverage(training)


def test_control_alignment_uses_exact_frame_grid_and_rejects_missing_rows():
    controls = [
        {"time_s": frame / 10 + 1e-13, "raw_action_rad": [0.] * 6,
         "applied_action_rad": [0.] * 6} for frame in range(31)
    ]
    assert control_indices(controls, 1530) == list(range(0, 1501, 50))
    with pytest.raises(ValueError, match="31 control"):
        control_indices(controls[:-1], 1530)
    with pytest.raises(ValueError, match="範囲外"):
        control_indices(controls, 1500)
    controls[5]["time_s"] = .51
    with pytest.raises(ValueError, match="時刻"):
        control_indices(controls, 1530)
    controls[5]["time_s"] = .5
    controls[5]["raw_action_rad"][0] = np.nan
    with pytest.raises(ValueError, match="action"):
        control_indices(controls, 1530)


def test_metrics_distinguish_draws_observations_raw_and_clipped_errors():
    records = [record(action=[.1, 0, 0, 0, 0, 0]), record(draw=1, action=[.04] * 6)]
    result = action_metrics(records)
    assert result["observation_count"] == 1
    assert result["raw"]["sample_count"] == 2
    assert result["raw"]["mae_rad"] == pytest.approx(.34 / 12)
    assert result["all_joints_within_0_05_rad_fraction"] == .5
    assert result["clipped_decision_fraction"] == 1
    assert result["bounded_estimate"]["mae_rad"] == 0
    assert result["policy_call_ms"]["median"] == 2
    with pytest.raises(ValueError, match="空"):
        action_metrics([])
    records[0]["action_rad"][0] = np.nan
    with pytest.raises(ValueError, match="非有限値"):
        action_metrics(records)


def test_pairing_requires_identical_noise_and_keeps_initial_difference_zero():
    records = [
        record(), record("saved_policy"), record(frame=1),
        record("saved_policy", frame=1, action=[.1, 0, 0, 0, 0, 0]),
    ]
    result = paired_effect(records)
    assert result["sample_count"] == 2
    assert result["mae_rad"] == pytest.approx(.1 / 12)
    records[-1]["noise_sha256"] = ["different"]
    with pytest.raises(ValueError, match="noise"):
        paired_effect(records)
    with pytest.raises(ValueError, match="サンプル数"):
        paired_effect(records[:1])


def test_error_reference_includes_boundary_without_tolerance_relaxation():
    records = [
        record(action=[.05] * 6),
        record(draw=1, action=[np.nextafter(.05, np.inf)] * 6),
    ]
    assert action_metrics(records)["all_joints_within_0_05_rad_fraction"] == .5


def test_csv_records_preserve_full_precision_arrays_and_seed_lists():
    original = record(action=[1 / 3] * 6)
    serialized = inference_comparison._csv_record(original)
    np.testing.assert_array_equal(json.loads(serialized["action_rad"]), original["action_rad"])
    assert json.loads(serialized["seeds"]) == original["seeds"]
    assert serialized["policy_call_ms"] == original["policy_call_ms"]


def test_cli_records_errors_without_a_success_report_or_input_changes(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    output = tmp_path / "outputs"

    def fail(*args):
        raise ValueError("input mismatch")

    monkeypatch.setattr(inference_comparison, "run_comparison", fail)
    result = inference_comparison.main([
        "--study", str(source), "--training", str(source), "--evaluation", str(tmp_path),
        "--reference-training", str(source), "--output-dir", str(output),
    ])
    assert result == 2
    directory, = output.iterdir()
    error = json.loads((directory / "error.json").read_text(encoding="utf-8"))
    assert error["status"] == "error"
    assert error["error_type"] == "ValueError"
    assert error["message"] == "input mismatch"
    assert not (directory / "comparison.json").exists()
    assert source.read_text(encoding="utf-8") == "{}"
    assert "input mismatch" in capsys.readouterr().err
