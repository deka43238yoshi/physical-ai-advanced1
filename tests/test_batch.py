from dataclasses import replace
import csv
import json

import numpy as np
import pytest

from so101_dice_sim.batch import (
    POSITION_BOUNDS_M, random_orientations, random_positions, run_batch,
)
from so101_dice_sim.cli import main


def test_random_positions_reproducible_and_bounded():
    positions = random_positions(100, 42)
    assert positions == random_positions(100, 42)
    assert positions != random_positions(100, 43)
    assert len(set(positions)) == 100
    for axis, (low, high) in enumerate(POSITION_BOUNDS_M):
        assert all(low <= position[axis] <= high for position in positions)
        assert len({position[axis] for position in positions}) == 100


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        random_positions(count, 42)


def test_batch_reports_and_keeps_orientation(tmp_path, settings):
    summary = run_batch(tmp_path, settings, 2, 42)
    assert summary["passed"]
    assert summary["passed_count"] == 2
    metadata = json.loads((tmp_path / "batch_settings.json").read_text())
    assert metadata["seed"] == 42
    assert metadata["initial_positions_m"] == [
        list(position) for position in random_positions(2, 42)
    ]
    with (tmp_path / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for index, position in enumerate(random_positions(2, 42), start=1):
        directory = tmp_path / f"run-{index:02d}"
        report = json.loads((directory / "result.json").read_text())
        assert report["settings"]["initial_position_m"] == list(position)
        assert report["settings"]["initial_quaternion_wxyz"] == list(
            settings.initial_quaternion_wxyz
        )
        with (directory / "trajectory.csv").open(newline="") as stream:
            samples = list(csv.DictReader(stream))
        assert len(samples) == 2501
        np.testing.assert_allclose(
            [float(samples[0][f"position_{axis}"]) for axis in "xyz"], position
        )
        np.testing.assert_allclose(
            [float(samples[0][f"quaternion_{axis}"]) for axis in "wxyz"],
            settings.initial_quaternion_wxyz,
        )
        assert report["validation"]["passed"]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_physics_failure_continues_batch(tmp_path, settings):
    short = replace(settings, duration_s=0.004, settle_window_s=0.002)
    summary = run_batch(tmp_path, short, 2, 42)
    assert not summary["passed"]
    assert summary["passed_count"] == 0
    assert len(summary["results"]) == 2
    assert all("table_contact" in row["failure_reasons"] for row in summary["results"])
    assert (tmp_path / "run-02" / "result.json").is_file()


def test_random_batch_cli(tmp_path):
    assert main([
        "--output-dir", str(tmp_path),
        "--random-position-runs", "2", "--seed", "42",
    ]) == 0
    first = next(tmp_path.iterdir())
    original = (first / "summary.json").read_bytes()
    assert main([
        "--output-dir", str(tmp_path),
        "--random-position-runs", "2", "--seed", "43",
    ]) == 0
    assert len(list(tmp_path.iterdir())) == 2
    assert (first / "summary.json").read_bytes() == original


def test_invalid_batch_cli(tmp_path, capsys):
    assert main([
        "--output-dir", str(tmp_path), "--random-position-runs", "0",
    ]) == 2
    assert "実行エラー" in capsys.readouterr().err


def test_random_orientations_reproducible_unit_and_isotropic():
    quaternions = random_orientations(20000, 42)
    assert quaternions == random_orientations(20000, 42)
    assert quaternions[:10] != random_orientations(10, 43)
    q = np.array(quaternions)
    assert np.isfinite(q).all()
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(np.mean(q, axis=0), 0, atol=0.02)
    np.testing.assert_allclose(q.T @ q / len(q), np.eye(4) / 4, atol=0.02)
    w, x, y, z = q.T
    normals = np.column_stack((
        2 * (x * z + w * y), 2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ))
    np.testing.assert_allclose(normals.mean(axis=0), 0, atol=0.02)
    np.testing.assert_allclose(
        normals.T @ normals / len(q), np.eye(3) / 3, atol=0.02
    )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_orientation_count(count):
    with pytest.raises(ValueError):
        random_orientations(count, 42)


def test_random_pose_batch(tmp_path, settings):
    summary = run_batch(tmp_path, settings, 3, 42, randomize_orientation=True)
    assert summary["randomize_orientation"]
    metadata = json.loads((tmp_path / "batch_settings.json").read_text())
    assert metadata["orientation_distribution"] == "uniform_so3"
    expected = random_orientations(3, 42)
    assert metadata["initial_quaternions_wxyz"] == [list(q) for q in expected]
    assert metadata["initial_positions_m"] == [
        list(p) for p in random_positions(3, 42)
    ]
    for index, quaternion in enumerate(expected, start=1):
        directory = tmp_path / f"run-{index:02d}"
        report = json.loads((directory / "result.json").read_text())
        assert report["settings"]["initial_quaternion_wxyz"] == list(quaternion)
        with (directory / "trajectory.csv").open(newline="") as stream:
            sample = next(csv.DictReader(stream))
        np.testing.assert_allclose(
            [float(sample[f"quaternion_{axis}"]) for axis in "wxyz"],
            quaternion, atol=1e-14,
        )
        row = summary["results"][index - 1]
        assert [row[f"initial_quaternion_{axis}"] for axis in "wxyz"] == list(
            quaternion
        )
        assert report["validation"]["passed"]
    assert summary["passed"]


def test_random_orientation_cli(tmp_path):
    assert main([
        "--output-dir", str(tmp_path), "--random-position-runs", "2",
        "--random-orientation", "--seed", "42",
    ]) == 0
    report = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert report["randomize_orientation"]


def test_orientation_requires_batch(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--random-orientation", "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
