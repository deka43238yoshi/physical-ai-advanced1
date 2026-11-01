import json

import numpy as np
import pytest

from so101_dice_sim.error_study import error_cases, main
from so101_dice_sim.grasp import run_grasp


def test_error_cases_have_exact_radius_and_eight_directions():
    cases = error_cases(3.1)
    assert cases[0] == (0, 0) and len(cases) == 9
    np.testing.assert_allclose(np.linalg.norm(cases[1:], axis=1), .0031, atol=1e-15)
    np.testing.assert_allclose(np.sum(cases[1:], axis=0), [0, 0], atol=1e-15)


@pytest.mark.parametrize("radius", [0, -1, 5.1, float("nan"), float("inf")])
def test_invalid_radius(radius):
    with pytest.raises(ValueError):
        error_cases(radius)


def test_injection_requires_observation_and_valid_error(tmp_path):
    with pytest.raises(ValueError, match="誤差注入"):
        run_grasp(tmp_path, place_face=1, grasp_error_m=(.0031, 0))
    with pytest.raises(ValueError):
        run_grasp(tmp_path, place_face=1, observe_position=True, grasp_error_m=(float("nan"), 0))
    assert not list(tmp_path.iterdir())


def test_error_only_changes_observed_grasp_target(tmp_path, monkeypatch):
    seen = []

    def stop_after_target(model, target, **kwargs):
        seen.append(target.copy())
        raise ValueError("テスト用IK停止")

    monkeypatch.setattr("so101_dice_sim.grasp.solve_approach", stop_after_target)
    with pytest.raises(ValueError, match="テスト用IK停止"):
        run_grasp(tmp_path, place_face=2, observe_position=True, grasp_error_m=(.0031, 0))
    plan = json.loads((tmp_path / "observation_plan.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(plan["observed_dice_center_m"][:2], [.245, .015])
    np.testing.assert_allclose(plan["unbiased_grasp_reference_xy_m"], [.25, 0], atol=1e-10)
    np.testing.assert_allclose(plan["grasp_reference_xy_m"], [.2531, 0], atol=1e-10)
    np.testing.assert_allclose(seen[0], [.2531, 0, .060], atol=1e-10)
    abort = json.loads((tmp_path / "placement_abort.json").read_text(encoding="utf-8"))
    assert not abort["release_executed"] and abort["last_time_s"] == pytest.approx(2)


def test_study_preserves_failed_cases_and_injection(tmp_path, monkeypatch):
    seen = []

    def fake_batch(directory, offsets, *, seed, grasp_error_m):
        seen.append((offsets, grasp_error_m))
        return {"passed_count": 0, "results": [
            {"target_face": face, "passed": False, "status": "execution_error",
             "failure_reason": "hold_speeds", "run_directory": f"face-{face}"}
            for face in (1, 2, 3)
        ]}

    monkeypatch.setattr("so101_dice_sim.error_study.run_position_batch", fake_batch)
    assert main(["--output-dir", str(tmp_path)]) == 1
    report = json.loads(next(tmp_path.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert report["runs"] == 27 and report["passed_count"] == 0
    assert len(seen) == 9
    assert all(offsets == [(0, 0)] for offsets, error in seen)
