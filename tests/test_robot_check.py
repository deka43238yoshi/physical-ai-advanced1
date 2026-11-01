import copy
import csv
import json

import numpy as np
import pytest

from so101_dice_sim.cli import main
from so101_dice_sim.robot_check import (
    JOINT_NAMES, ROOT, SOURCE_COMMIT, run_robot_check, validate_robot_samples,
    verify_assets,
)


@pytest.fixture(scope="module")
def robot_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("so101")
    return directory, run_robot_check(directory)


def test_assets_are_pinned_and_unmodified():
    provenance = verify_assets(ROOT / "assets" / "robotstudio_so101")
    assert provenance["commit"] == SOURCE_COMMIT
    assert provenance["license"] == "Apache-2.0"
    assert len(provenance["files"]) == 23
    assert provenance["modified_upstream_files"] == []
    assert any(entry["path"] == "LICENSE" for entry in provenance["files"])


def test_asset_corruption_rejected(tmp_path):
    (tmp_path / "model.xml").write_text("changed")
    (tmp_path / "provenance.json").write_text(json.dumps({
        "commit": SOURCE_COMMIT,
        "files": [{"path": "model.xml", "sha256": "incorrect"}],
    }))
    with pytest.raises(ValueError, match="ハッシュ"):
        verify_assets(tmp_path)


def test_robot_structure_hold_motion_and_open_close(robot_run):
    _, report = robot_run
    assert report["validation"]["passed"], report["validation"]
    assert all(report["validation"]["checks"].values())
    assert [joint["name"] for joint in report["joints"]] == list(JOINT_NAMES)
    assert report["timestep_s"] == 0.002
    assert report["duration_s"] == pytest.approx(11)
    assert report["sample_count"] == 5500
    assert not report["initial_contacts"]
    assert not report["warnings"]
    assert report["validation"]["max_contact_count"] == 0
    assert report["validation"]["max_settled_tracking_error_rad"] < 0.01
    assert report["validation"]["max_settled_speed_rad_s"] < 0.01
    phases = {phase["phase"]: phase for phase in report["phases"]}
    assert phases["open"]["actual_rad"][5] == pytest.approx(1, abs=0.01)
    assert phases["close"]["actual_rad"][5] == pytest.approx(0, abs=0.01)
    assert phases["open"]["tip_surface_distance_m"] > (
        phases["close"]["tip_surface_distance_m"] + 0.03
    )
    assert all(j["forcerange_nm"] == [-2.94, 2.94] for j in report["joints"])


def test_robot_csv_matches_report(robot_run):
    directory, report = robot_run
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == report
    with (directory / "trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5500
    for index, row in enumerate(rows, start=1):
        assert float(row["time_s"]) == pytest.approx(index * 0.002, abs=1e-9)
        assert int(row["contact_count"]) == 0
        for joint in report["joints"]:
            value = float(row[f"{joint['name']}_qpos"])
            low, high = joint["range_rad"]
            assert low - 1e-6 <= value <= high + 1e-6
    np.testing.assert_allclose(
        [float(rows[-1][f"{name}_qpos"]) for name in JOINT_NAMES],
        report["phases"][-1]["actual_rad"],
    )
    with pytest.raises(FileExistsError):
        run_robot_check(directory)


def valid_samples():
    return [
        {
            "qpos": [0] * 6, "qvel": [0] * 6, "force": [0] * 6,
            "target": [0] * 6, "settled": False, "contact_count": 0,
            "tip_surface_distance_m": 0.02,
        },
        {
            "qpos": [0.1] * 5 + [1], "qvel": [0] * 6, "force": [0] * 6,
            "target": [0.1] * 5 + [1], "settled": True, "contact_count": 0,
            "tip_surface_distance_m": 0.08,
        },
        {
            "qpos": [0.1] * 5 + [0], "qvel": [0] * 6, "force": [0] * 6,
            "target": [0.1] * 5 + [0], "settled": True, "contact_count": 0,
            "tip_surface_distance_m": 0.02,
        },
    ]


@pytest.mark.parametrize("kind,check", [
    ("contact", "no_contacts"),
    ("position", "joint_limits"),
    ("force", "force_limits"),
    ("speed", "settled_speed"),
    ("tracking", "tracking"),
    ("finite", "finite_values"),
    ("warning", "no_mujoco_warnings"),
    ("no_motion", "all_arm_joints_moved"),
    ("not_closed", "gripper_opened_and_closed"),
])
def test_robot_validation_detects_failures(kind, check):
    samples = copy.deepcopy(valid_samples())
    warnings = 0
    if kind == "contact":
        samples[0]["contact_count"] = 1
    elif kind == "position":
        samples[0]["qpos"][0] = 4
    elif kind == "force":
        samples[0]["force"][0] = 3
    elif kind == "speed":
        samples[1]["qvel"][0] = 0.01
    elif kind == "tracking":
        samples[1]["qpos"][0] = 0.2
    elif kind == "finite":
        samples[0]["qpos"][0] = float("nan")
    elif kind == "warning":
        warnings = 1
    elif kind == "no_motion":
        for sample in samples:
            sample["qpos"][0] = 0
    elif kind == "not_closed":
        samples[-1]["qpos"][5] = 0.5
    result = validate_robot_samples(
        samples, np.tile([-2, 2], (6, 1)),
        np.tile([-2.94, 2.94], (6, 1)), warnings,
    )
    assert not result["passed"]
    assert check in result["failure_reasons"]


def test_robot_cli(tmp_path):
    assert main(["--check-so101", "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["task"] == "so101_standalone_check"
    assert report["validation"]["passed"]


def test_robot_cli_rejects_dice_batch():
    with pytest.raises(SystemExit) as error:
        main(["--check-so101", "--random-position-runs", "2"])
    assert error.value.code == 2
