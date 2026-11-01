import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from so101_dice_sim.cli import main
from so101_dice_sim.reporting import create_run_directory, save_result
from so101_dice_sim.simulation import validate_result


ROOT = Path(__file__).resolve().parents[1]


def test_output_schema_and_values(tmp_path, settings, settled):
    directory = create_run_directory(tmp_path)
    save_result(directory, settings, settled, validate_result(settled, settings))
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["validation"]["passed"]
    assert report["sample_count"] == 2501
    assert report["steps"] == 2500
    assert report["units"]["quaternion"] == "wxyz"
    assert report["units"]["vectors_frame"] == "world"
    assert report["versions"]["mujoco"] == "3.3.7"
    assert report["face_axes_local"]["1"] == [0, 0, 1]
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2501
    assert rows[0]["top_face"] == ""
    for row, sample in zip(rows, settled.samples):
        assert float(row["time_s"]) == sample.time_s
        assert float(row["position_z"]) == sample.position_m[2]
        assert float(row["quaternion_w"]) == sample.quaternion_wxyz[0]
        assert float(row["linear_velocity_z"]) == sample.linear_velocity_m_s[2]
        assert float(row["angular_velocity_x"]) == sample.angular_velocity_rad_s[0]
        assert row["table_contact"] == str(sample.table_contact)
        assert row["top_face"] == (
            "" if sample.top_face is None else str(sample.top_face)
        )
    assert float(rows[-1]["time_s"]) == pytest.approx(5)
    before = (directory / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_result(directory, settings, settled, validate_result(settled, settings))
    assert (directory / "result.json").read_bytes() == before
    assert create_run_directory(tmp_path) != directory


def test_cli_success_and_no_overwrite(tmp_path):
    args = ["--output-dir", str(tmp_path)]
    assert main(args) == 0
    first = next(tmp_path.iterdir())
    original = (first / "result.json").read_bytes()
    assert main(args) == 0
    assert len(list(tmp_path.iterdir())) == 2
    assert (first / "result.json").read_bytes() == original


def test_cli_failed_physics_persists_results(tmp_path, capsys):
    config = json.loads((ROOT / "config" / "default.json").read_text())
    config.update(
        scene=str(ROOT / "assets" / "scene.xml"),
        duration_s=0.004,
        settle_window_s=0.002,
    )
    path = tmp_path / "short.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    outputs = tmp_path / "outputs"
    assert main(["--config", str(path), "--output-dir", str(outputs)]) == 1
    assert "物理検証失敗" in capsys.readouterr().err
    directory = next(outputs.iterdir())
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert not report["validation"]["passed"]
    assert "table_contact" in report["validation"]["failure_reasons"]
    assert (directory / "trajectory.csv").is_file()


def test_cli_missing_config(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    assert main([
        "--config", str(tmp_path / "missing.json"),
        "--output-dir", str(outputs),
    ]) == 2
    assert "実行エラー" in capsys.readouterr().err
    error = next(outputs.glob("*/error.json"))
    assert json.loads(error.read_text(encoding="utf-8"))["status"] == "error"


def test_cli_output_permission_error(tmp_path, capsys):
    file = tmp_path / "not_a_directory"
    file.write_text("existing", encoding="utf-8")
    assert main(["--output-dir", str(file)]) == 2
    assert "実行エラー" in capsys.readouterr().err
    assert file.read_text(encoding="utf-8") == "existing"


def test_module_entrypoint_exit_code(tmp_path):
    completed = subprocess.run(
        [
            sys.executable, "-m", "so101_dice_sim",
            "--config", str(tmp_path / "missing.json"),
            "--output-dir", str(tmp_path / "outputs"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert "実行エラー" in completed.stderr
