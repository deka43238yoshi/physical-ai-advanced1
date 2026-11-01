import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .action_contract import CONTRACT_NAME, ENDPOINT_MODE, action_contract
from .grasp import validate_grasp
from .placement import validate_placement
from .reporting import write_json
from .robot_check import JOINT_NAMES, ROOT, verify_assets
from .rotation import validate_rotation
from .visualization import Replay


REPO_ID = "local/so101_face2_endpoint_v2"


def load_endpoint_replay(source: Path) -> tuple[Replay, np.ndarray]:
    from .visualization import load_replay

    verify_assets(ROOT / "assets" / "robotstudio_so101")
    report = json.loads((source / "replay.json").read_text(encoding="utf-8"))
    if (
        report["mode"] != "endpoint_linear_bounded" or not report["passed"]
        or not report["completed"] or report["requested_face"] != 2
        or report["physics_steps"] != 22500 or report["warnings"]
    ):
        raise ValueError("安全制限付き線形補間で成功した面2の実軌道が必要です")
    original = load_replay(Path(report["source"]))
    if not np.array_equal(report["initial_qpos"], original.report["initial_qpos"]):
        raise ValueError("元教師と実軌道の初期状態が一致しません")
    path = source / "trajectory.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["trajectory_sha256"]:
        raise ValueError("補間実軌道のハッシュが一致しません")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    samples = [{
        key: value if key == "phase" else value == "True" if value in ("True", "False")
        else None if value == "" else int(value) if key == "top_face" else float(value)
        for key, value in row.items()
    } for row in rows]
    if len(samples) != 22501:
        raise ValueError("補間実軌道のサンプル数が一致しません")
    if not all((
        validate_grasp(samples[:9501], {}, observe_position=True)["passed"],
        validate_rotation(samples[:12501], {}, target_face=2, observe_position=True)["passed"],
        validate_placement(samples, {}, target_face=2, observe_position=True)["passed"],
    )):
        raise ValueError("補間実軌道が既存の物理成功判定を満たしません")
    times = np.array([s["time_s"] for s in samples])
    columns = [
        *(f"{name}_qpos" for name in JOINT_NAMES),
        *(f"dice_{axis}_m" for axis in "xyz"),
        *(f"dice_quat_{axis}" for axis in "wxyz"),
    ]
    qpos = np.array([[s[key] for key in columns] for s in samples])
    targets = np.array([[s[f"{name}_target"] for name in JOINT_NAMES] for s in samples])
    if not np.isfinite(qpos).all() or not np.allclose(
        np.linalg.norm(qpos[:, 9:13], axis=1), 1, atol=1e-8, rtol=0,
    ):
        raise ValueError("実軌道の状態または姿勢が不正です")
    if not np.allclose(qpos[0], report["initial_qpos"], atol=1e-12, rtol=0):
        raise ValueError("実軌道CSVと報告の初期状態が一致しません")
    controls = report["controls"]
    if len(controls) != 450:
        raise ValueError("補間実軌道に450周期の末端目標がありません")
    for index, control in enumerate(controls):
        start = index * 50
        end = start + 50
        endpoint = np.array(control["applied_action_rad"])
        if (
            control["teacher_endpoint_row"] != end
            or not np.isclose(control["target_time_s"], end * .002, atol=1e-9, rtol=0)
            or not np.array_equal(endpoint, targets[end])
            or not np.array_equal(control["start_action_rad"], targets[start])
        ):
            raise ValueError("実軌道の末端目標時刻・指令が一致しません")
        expected = targets[start] + np.arange(1, 51)[:, None] / 50 * (endpoint - targets[start])
        expected[-1] = endpoint
        if not np.allclose(targets[start + 1:end + 1], expected, atol=1e-14, rtol=0):
            raise ValueError("実軌道が線形補間ではありません")
    return Replay(report, rows, times, qpos, {}), targets[np.arange(50, 22501, 50)].astype(np.float32)


def collect_endpoints(directory: Path, sources: list[Path]) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .learning import CAMERAS, FPS, TASK, render_observation
    from .visualization import ReplayRenderer

    if not sources or len({source.resolve() for source in sources}) != len(sources):
        raise ValueError("重複しない補間実軌道を指定してください")
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        **{key: {"dtype": "image", "shape": (256, 256, 3),
                 "names": ["height", "width", "channels"]} for key in CAMERAS},
    }
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, root=directory / "dataset", fps=FPS, robot_type="so101_mujoco_rad",
        features=features, use_videos=False,
    )
    contract = action_contract(ENDPOINT_MODE)
    write_json(directory / "dataset" / CONTRACT_NAME, contract)
    renderer = ReplayRenderer()
    episodes = []
    try:
        for source in sources:
            replay, actions = load_endpoint_replay(source)
            for frame, index in enumerate(range(0, 22500, 50)):
                dataset.add_frame({
                    "observation.state": replay.qpos[index, :6].astype(np.float32),
                    "action": actions[frame],
                    **render_observation(renderer, replay.qpos[index]), "task": TASK,
                })
            dataset.save_episode()
            episodes.append({
                "source": str(source.resolve()), "frames": len(actions),
                "source_sha256": {
                    name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                    for name in ("replay.json", "trajectory.csv")
                },
            })
            print(f"次周期末教師保存: {source.name}", flush=True)
        dataset.finalize()
    finally:
        renderer.close()
    report = {
        "dataset": str((directory / "dataset").resolve()), "repo_id": REPO_ID,
        "episodes": episodes, "action_contract": contract, "fps": FPS, "joint_units": "rad",
        "cameras": CAMERAS, "task": TASK, "uploaded": False,
        "notice": "安全補正後の実状態画像・関節から次0.1秒末の適用目標を学習する別データ。少数接続試験。",
    }
    write_json(directory / "collection.json", report)
    return report
