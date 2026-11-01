import csv
import hashlib
import json
import subprocess

import mujoco
import numpy as np
import pytest

from so101_dice_sim import comparison_visualization as comparison
from so101_dice_sim import visualization as viz


def save_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    success = tmp_path / "success"
    failure = tmp_path / "failure"
    success.mkdir()
    failure.mkdir()
    row = {
        "time_s": 0, "phase": "policy", "top_face": "1",
        **dict.fromkeys(viz.METRICS, 0), "unexpected_contacts": 0,
        **{f"{name}_qpos": 0 for name in viz.JOINT_NAMES},
        "dice_x_m": .2425, "dice_y_m": .0125, "dice_z_m": .016,
        "dice_quat_w": 1, "dice_quat_x": 0, "dice_quat_y": 0, "dice_quat_z": 0,
    }
    rows = [{**row, "time_s": index * .002} for index in range(3)]
    columns = [
        *(f"{name}_qpos" for name in viz.JOINT_NAMES),
        *(f"dice_{axis}_m" for axis in "xyz"),
        *(f"dice_quat_{axis}" for axis in "wxyz"),
    ]
    teacher_rows = [{**sample, "phase": "settle"} for sample in rows]
    teacher = viz.Replay(
        {"requested_face": 2}, teacher_rows, np.array([0, .002, .004]),
        np.array([[sample[key] for key in columns] for sample in rows]), {},
    )
    monkeypatch.setattr(comparison, "load_endpoint_replay", lambda source: (teacher, None))
    rows[-1]["unexpected_contacts"] = 2
    save_rows(failure / "trajectory.csv", rows)
    save_rows(success / "trajectory.csv", teacher_rows)
    (success / "replay.json").write_text("{}", encoding="utf-8")
    (failure / "evaluation.json").write_text(json.dumps({
        "passed": False, "completed": False, "physics_steps": 2, "duration_s": .004,
        "requested_face": 2,
        "offset_mm": [-2.5, -2.5], "failure_reasons": ["unexpected_contacts:[(0,31)]"],
        "final_state": rows[-1],
    }), encoding="utf-8")
    (tmp_path / "settings.json").write_text(json.dumps({
        "split": {"target_face": 2, "evaluation_offsets_mm": [[-2.5, -2.5]]},
    }), encoding="utf-8")
    return success, failure


def test_load_comparison(sources):
    teacher, failed = comparison.load_comparison(*sources)
    np.testing.assert_array_equal(teacher.qpos[0], failed.qpos[0])
    assert teacher.report["validation"]["passed"] is True
    assert failed.report["validation"]["passed"] is False
    assert failed.report["task"] == "smolvla_evaluation"
    assert viz.replay_timing(failed) is None
    assert viz.phase_label("policy", 2) == "SmolVLAによる制御"
    indices = viz.frame_indices(failed.times, 12)
    assert indices[0] == 0 and indices[-1] == len(failed.times) - 1


def test_load_completed_policy_as_successful_demo(sources):
    success, policy_path = sources
    trajectory = policy_path / "trajectory.csv"
    with trajectory.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["unexpected_contacts"] = "0"
    save_rows(trajectory, rows)
    report_path = policy_path / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(passed=True, completed=True, failure_reasons=[])
    report["final_state"] = {
        key: value if key in ("phase", "top_face") else (
            value == "True" if value in ("True", "False") else float(value)
        )
        for key, value in rows[-1].items()
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    teacher, policy = comparison.load_comparison(success, policy_path)
    assert teacher.report["validation"]["passed"] is True
    assert policy.report["validation"]["passed"] is True
    assert policy.report["display_label"] == "SmolVLA（最終結果：面2配置成功）"


@pytest.mark.parametrize("key,value", [
    ("time_s", "nan"), ("time_s", ".01"), ("dice_quat_w", "2"),
    ("dice_x_m", "inf"), ("phase", "unknown"), ("top_face", "8"),
    ("unexpected_contacts", "1"),
])
def test_reject_bad_csv(sources, key, value):
    success, failure = sources
    path = failure / "trajectory.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1][key] = value
    save_rows(path, rows)
    with pytest.raises(ValueError):
        comparison.load_comparison(success, failure)


@pytest.mark.parametrize("key,value", [
    ("passed", True), ("passed", 0), ("completed", "true"),
    ("duration_s", 42), ("physics_steps", 3), ("offset_mm", [5, 5]),
    ("failure_reasons", []),
])
def test_reject_bad_report(sources, key, value):
    success, failure = sources
    path = failure / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[key] = value
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        comparison.load_comparison(success, failure)


def test_reject_different_initial_state(sources):
    success, failure = sources
    path = failure / "trajectory.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["dice_x_m"] = ".245"
    save_rows(path, rows)
    with pytest.raises(ValueError):
        comparison.load_comparison(success, failure)


def test_reject_final_state_mismatch_and_cli_error(sources, tmp_path):
    success, failure = sources
    path = failure / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["final_state"]["dice_x_m"] = .9
    path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "visual"
    assert comparison.main([
        "--success", str(success), "--failure", str(failure), "--output-dir", str(output),
    ]) == 2
    directory, = output.iterdir()
    assert "最終状態" in json.loads((directory / "error.json").read_text(encoding="utf-8"))["message"]
    assert not (directory / "comparison.json").exists()


@pytest.mark.parametrize("passed", [False, True])
def test_render_and_encode_comparison(sources, tmp_path, monkeypatch, passed):
    from PIL import Image

    def forbid_step(*args, **kwargs):
        pytest.fail("描画時に物理ステップを実行しました")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    success, failure = sources
    if passed:
        path = failure / "trajectory.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["unexpected_contacts"] = "0"
        save_rows(path, rows)
        report_path = failure / "evaluation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(passed=True, completed=True, failure_reasons=[])
        report["final_state"] = {
            key: value if key in ("phase", "top_face") else (
                value == "True" if value in ("True", "False") else float(value)
            )
            for key, value in rows[-1].items()
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
    originals = {
        path: path.read_bytes()
        for directory in sources for path in directory.iterdir()
    }
    output = tmp_path / "visual"
    assert comparison.main([
        "--success", str(success), "--failure", str(failure),
        "--output-dir", str(output), "--fps", "2",
    ]) == 0
    directory, = output.iterdir()
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    assert report["policy_passed"] is passed
    assert report["comparison_frame_count"] == 2 and report["physics_steps"] == 0
    for label, source in (("success", success), ("policy", failure)):
        for name, digest in report["source_sha256"][label].items():
            assert digest == hashlib.sha256((source / name).read_bytes()).hexdigest()
    last_image = "final_comparison.png" if passed else "failure_instant.png"
    for filename in ("comparison_sheet.jpg", last_image, "comparison_metrics.png"):
        with Image.open(directory / filename) as image:
            assert image.width >= 1280 and np.asarray(image).std() > 10
    for name in ("comparison.mp4", "success_teacher/replay.mp4", "policy_smolvla/replay.mp4"):
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,pix_fmt", "-of", "json",
            str(directory / name),
        ], check=True, capture_output=True, text=True)
        video, = json.loads(probe.stdout)["streams"]
        assert video["nb_read_frames"] == "2" and video["pix_fmt"] == "yuv420p"
        assert video["width"] == 1280
        assert video["height"] == (400 if name == "comparison.mp4" else 576)
    for path, content in originals.items():
        assert path.read_bytes() == content
