import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest

from so101_dice_sim import visualization as viz
from so101_dice_sim import wrist_camera


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    report = {
        "task": "so101_fixed_place_face_2",
        "source_commit": viz.SOURCE_COMMIT,
        "mujoco_version": mujoco.__version__,
        "sample_count": 3, "timestep_s": .002, "duration_s": .004,
        "validation": {"passed": True},
    }
    (directory / "result.json").write_text(json.dumps(report), encoding="utf-8")
    row = {
        "time_s": 0, "phase": "initial", "top_face": 1,
        **dict.fromkeys(viz.METRICS, 0),
        **{f"{name}_qpos": 0 for name in viz.JOINT_NAMES},
        "dice_x_m": .245, "dice_y_m": .015, "dice_z_m": .016,
        "dice_quat_w": 1, "dice_quat_x": 0, "dice_quat_y": 0, "dice_quat_z": 0,
    }
    row["gripper_qpos"] = .6
    with (directory / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        for i in range(3):
            writer.writerow({**row, "time_s": i * .002})
    return directory


def test_load_replay(source):
    replay = viz.load_replay(source)
    assert replay.qpos.shape == (3, 13)
    assert replay.qpos[0, 5] == .6
    np.testing.assert_array_equal(replay.qpos[0, 6:9], [.245, .015, .016])
    np.testing.assert_array_equal(replay.qpos[0, 9:], [1, 0, 0, 0])


def test_face_one_replay_and_labels(source):
    path = source / "result.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(task="so101_fixed_place_face_1", requested_face=1)
    path.write_text(json.dumps(report), encoding="utf-8")
    assert viz.load_replay(source).report["requested_face"] == 1
    assert viz.phase_label("rotate", 1) == "姿勢維持（回転量0）"
    assert viz.phase_label("rotation_hold", 1) == "上面1で保持"
    assert viz.phase_label("rotation_hold", 2) == "上面2で保持"
    assert viz.phase_label("touch", 1) == viz.PHASES["touch"]


def test_face_three_replay_labels_and_time_windows(source):
    path = source / "result.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(task="so101_fixed_place_face_3", requested_face=3)
    path.write_text(json.dumps(report), encoding="utf-8")
    replay = viz.load_replay(source)
    assert viz.phase_label("wrist_twist", 3) == "手首ねじり−90度"
    assert viz.phase_label("rotation_hold", 3) == "上面3で保持"
    replay.times = np.array([0., 46.])
    timing = viz.replay_timing(replay)
    assert timing.rotation_hold_start_s == 24
    assert timing.rotation_end_s == 26
    assert timing.support_start_s == 34.5 and timing.support_end_s == 35
    assert timing.final_start_s == 45 and timing.end_s == 46
    assert len(viz.frame_indices(replay.times, 24)) == 1104
    replay.report["requested_face"] = 2
    assert viz.replay_timing(replay).final_start_s == 41


def test_observed_replay_time_windows(source):
    replay = viz.load_replay(source)
    replay.report.update(planning_mode="observed_position", requested_face=3)
    replay.times = np.array([0., 49.])
    timing = viz.replay_timing(replay)
    assert timing.grasp_end_s == 19
    assert timing.rotation_hold_start_s == 27 and timing.rotation_end_s == 29
    assert timing.support_start_s == 37.5 and timing.support_end_s == 38
    assert timing.final_start_s == 48 and timing.end_s == 49
    assert viz.phase_label("hold_settle", 3) == "把持後の静止待ち"


@pytest.mark.parametrize("field,value", [
    ("task", "unknown"), ("source_commit", "changed"),
    ("mujoco_version", "0"), ("sample_count", 4), ("timestep_s", -.002),
    ("duration_s", 42), ("duration_s", None), ("validation", {"passed": "true"}),
])
def test_reject_incompatible_report(source, field, value):
    path = source / "result.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        viz.load_replay(source)


@pytest.mark.parametrize("column,value", [
    ("time_s", "nan"), ("time_s", ".5"), ("dice_quat_w", "2"),
    ("dice_x_m", "inf"), ("phase", "unknown"), ("top_face", "8"),
    ("tilt_deg", "nan"),
])
def test_reject_bad_csv(source, column, value):
    path = source / "trajectory.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1][column] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        viz.load_replay(source)


def test_missing_column(source):
    path = source / "trajectory.csv"
    path.write_text("time_s,phase\n0,initial\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV"):
        viz.load_replay(source)


@pytest.mark.parametrize("fps", [0, -1, 61, True, 2.5])
def test_bad_fps(fps):
    with pytest.raises(ValueError):
        viz.frame_indices(np.array([0, 42]), fps)


def test_frame_schedule():
    times = np.arange(21001) * .002
    indices = viz.frame_indices(times, 24)
    assert len(indices) == 1008
    assert indices[0] == 0 and indices[-1] == 21000
    assert np.all(np.diff(indices) > 0)
    assert np.max(np.abs(times[indices] - np.linspace(0, 42, 1008))) <= .001


def test_pips_only_modify_render_scene():
    model = mujoco.MjModel.from_xml_path(str(viz.ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos = data.qpos.copy()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    before = model.ngeom
    viz.add_pips(scene, data.xpos[model.body("dice").id], np.eye(3))
    assert scene.ngeom == 21
    assert model.ngeom == before
    np.testing.assert_array_equal(data.qpos, qpos)


def test_cli_error(source, tmp_path, capsys):
    assert viz.main(["--run-dir", str(source), "--fps", "0"]) == 2
    assert "fps" in capsys.readouterr().err


HAS_RENDERING = (
    os.environ.get("MUJOCO_GL") == "osmesa"
    and importlib.util.find_spec("matplotlib") is not None
    and importlib.util.find_spec("PIL") is not None
)


@pytest.mark.skipif(not HAS_RENDERING, reason="専用visualize-testイメージで描画を検証")
def test_render_and_encode(source, tmp_path, monkeypatch):
    from PIL import Image

    def forbid_step(*args, **kwargs):
        pytest.fail("再生中に物理ステップを実行しました")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    originals = {name: (source / name).read_bytes() for name in ("result.json", "trajectory.csv")}
    output = tmp_path / "visual"
    assert viz.main(["--run-dir", str(source), "--output-dir", str(output), "--fps", "2"]) == 0
    directory, = output.iterdir()
    report = json.loads((directory / "visualization.json").read_text(encoding="utf-8"))
    assert report["physics_steps"] == 0 and report["frame_count"] == 2
    for name in ("final.png", "contact_sheet.jpg", "metrics.png"):
        with Image.open(directory / name) as image:
            assert image.width >= 1280
            assert np.asarray(image).std() > 10
    assert (directory / "metrics.svg").stat().st_size > 1000
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,pix_fmt",
        "-of", "json", str(directory / "replay.mp4"),
    ], check=True, capture_output=True, text=True)
    video = json.loads(probe.stdout)["streams"][0]
    assert (video["width"], video["height"], video["nb_read_frames"]) == (1280, 576, "2")
    assert video["pix_fmt"] == "yuv420p"
    for name, content in originals.items():
        assert (source / name).read_bytes() == content


@pytest.mark.skipif(not HAS_RENDERING, reason="専用visualize-testイメージで描画を検証")
def test_rendered_qpos_matches_record(source):
    replay = viz.load_replay(source)
    renderer = viz.ReplayRenderer()
    try:
        renderer.frame(replay, 2)
        np.testing.assert_array_equal(renderer.data.qpos, replay.qpos[2])
        assert renderer.data.time == 0
    finally:
        renderer.close()


@pytest.mark.parametrize("times", [[], [-1], [float("nan")], [float("inf")], [.005]])
def test_wrist_rejects_bad_times(source, times):
    with pytest.raises(ValueError, match="撮影時刻"):
        wrist_camera.capture_indices(viz.load_replay(source), times)


def test_wrist_nearest_sample(source):
    assert wrist_camera.capture_indices(viz.load_replay(source), [0, .0021, .004]) == [0, 1, 2]


def test_external_camera_and_initial_marker_are_world_fixed():
    model = mujoco.MjModel.from_xml_path(str(viz.ROOT / "assets" / "so101_dice_scene.xml"))
    camera = model.camera("external_overhead").id
    assert model.cam_bodyid[camera] == 0
    np.testing.assert_array_equal(model.cam_pos[camera], [0, -1, 1.4])
    assert model.cam_fovy[camera] == 50
    marks = [model.site(f"initial_mark_{side}").id for side in ("left", "right", "bottom", "top")]
    assert all(model.site_bodyid[s] == 0 and model.site_group[s] == 5 for s in marks)
    np.testing.assert_allclose(np.mean(model.site_pos[marks, :2], axis=0), [.245, .015])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    position, rotation = data.cam_xpos[camera].copy(), data.cam_xmat[camera].copy()
    data.qpos[0] = .2
    data.qpos[6:8] += [.005, -.005]
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.cam_xpos[camera], position)
    np.testing.assert_array_equal(data.cam_xmat[camera], rotation)
    focal = 240 / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
    for x in (-.6, .6):
        for y in (-.45, .45):
            point = rotation.reshape(3, 3).T @ (np.array([x, y, 0]) - position)
            assert point[2] < 0
            pixel = np.array([320, 240]) + focal * point[:2] / -point[2]
            assert 0 < pixel[0] < 640 and 0 < pixel[1] < 480


@pytest.mark.skipif(not HAS_RENDERING, reason="専用visualize-testイメージで描画を検証")
def test_external_camera_sees_initial_dice_and_marker(source, tmp_path):
    replay = viz.load_replay(source)
    report = wrist_camera.capture_wrist(
        replay, tmp_path, [0, .004], camera_name="external_overhead",
    )
    assert report["camera"] == "external_overhead"
    assert all(c["visible_dice_pixels"] >= 80 and not c["mask_touches_image_border"] for c in report["captures"])
    from PIL import Image
    with Image.open(tmp_path / report["captures"][0]["rgb_file"]) as image:
        rgb = np.asarray(image).astype(int)
        cyan = (rgb[:, :, 1] > rgb[:, :, 0] + 30) & (rgb[:, :, 2] > rgb[:, :, 0] + 30)
        assert np.count_nonzero(cyan) > 10


@pytest.mark.skipif(not HAS_RENDERING, reason="専用visualize-testイメージで描画を検証")
def test_wrist_capture_uses_named_camera_without_physics(source, tmp_path, monkeypatch):
    from PIL import Image

    def forbid_step(*args, **kwargs):
        pytest.fail("手首カメラ再生中に物理ステップを実行しました")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    original = {name: (source / name).read_bytes() for name in ("result.json", "trajectory.csv")}
    output = tmp_path / "camera"
    assert wrist_camera.main([
        "--run-dir", str(source), "--output-dir", str(output), "--times", "0", ".004",
    ]) == 0
    directory, = output.iterdir()
    report = json.loads((directory / "camera_report.json").read_text(encoding="utf-8"))
    assert report["camera"] == "wrist_cam" and report["physics_steps"] == 0
    assert report["vertical_fov_deg"] == 61
    assert len(report["captures"]) == 2
    model = mujoco.MjModel.from_xml_path(str(viz.ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    replay = viz.load_replay(source)
    for capture in report["captures"]:
        data.qpos[:] = replay.qpos[capture["sample_index"]]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(capture["camera_position_world_m"], data.cam_xpos[model.camera("wrist_cam").id])
        with Image.open(directory / capture["rgb_file"]) as rgb:
            assert rgb.size == (640, 480) and np.asarray(rgb).std() > 5
        with Image.open(directory / capture["mask_file"]) as mask:
            pixels = np.asarray(mask)
            assert set(np.unique(pixels)).issubset({0, 255})
            assert np.count_nonzero(pixels) == capture["visible_dice_pixels"]
    for name, content in original.items():
        assert (source / name).read_bytes() == content
