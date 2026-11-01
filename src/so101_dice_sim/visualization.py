import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from .faces import FACE_AXES
from .placement import PlacementTiming, placement_timing
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES, ROOT, SOURCE_COMMIT, verify_assets


FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PHASES = {
    "policy": "SmolVLAによる制御",
    "initial": "初期状態", "settle": "着地待ち", "pregrasp": "把持準備",
    "lower": "下降", "close": "閉指", "lift": "持ち上げ", "hold": "保持",
    "hold_settle": "把持後の静止待ち",
    "rotate": "回転", "rotation_hold": "上面2で保持",
    "wrist_twist": "手首ねじり−90度",
    "place_above": "配置位置の上へ", "touch": "接地・支持確認",
    "release": "開指", "retreat": "退避", "final_settle": "静止待ち",
    "final_evaluation": "最終静止評価",
}
PIPS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    5: [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)],
    6: [(x, y) for x in (-1, 1) for y in (-1, 0, 1)],
}
METRICS = (
    "dice_bottom_gap_m", "dice_linear_speed_m_s", "dice_angular_speed_rad_s",
    "fixed_jaw_force_n", "moving_jaw_force_n", "dice_table_force_n",
    "tilt_deg", "gripper_qpos",
)


def phase_label(phase: str, requested_face: int | None) -> str:
    if requested_face == 1 and phase in ("rotate", "rotation_hold"):
        return "姿勢維持（回転量0）" if phase == "rotate" else "上面1で保持"
    if phase == "rotation_hold" and requested_face is not None:
        return f"上面{requested_face}で保持"
    return PHASES[phase]


@dataclass
class Replay:
    report: dict
    rows: list[dict[str, str]]
    times: np.ndarray
    qpos: np.ndarray
    values: dict[str, np.ndarray]


def replay_timing(replay: Replay) -> PlacementTiming | None:
    if not replay.report["task"].startswith("so101_fixed_place_face_"):
        return None
    timing = placement_timing(
        replay.report.get("requested_face", 2),
        observe_position=replay.report.get("planning_mode") == "observed_position",
    )
    return timing if replay.times[-1] >= timing.end_s - 1e-9 else None


def load_replay(source: Path) -> Replay:
    with (source / "result.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("result.jsonはオブジェクトである必要があります")
    if report.get("task") not in {
        "so101_fixed_place_face_1", "so101_fixed_place_face_2",
        "so101_fixed_place_face_3",
        "so101_fixed_grasp_rotate", "so101_fixed_grasp_lift",
    } or report.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("可視化は同じSO-101資産の把持・回転・配置結果に対応します")
    if report.get("mujoco_version") != mujoco.__version__:
        raise ValueError("保存結果と描画用MuJoCoのバージョンが一致しません")
    duration = report.get("duration_s")
    validation = report.get("validation")
    if (
        not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
        or not isinstance(validation, dict)
        or not isinstance(validation.get("passed"), bool)
    ):
        raise ValueError("保存結果の実行時間または検証結果が不正です")
    with (source / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "time_s", "phase", "top_face", *METRICS,
            *(f"{name}_qpos" for name in JOINT_NAMES),
            *(f"dice_{axis}_m" for axis in "xyz"),
            *(f"dice_quat_{axis}" for axis in "wxyz"),
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("再生に必要なCSV列がありません")
        rows = list(reader)
    if len(rows) < 2 or len(rows) != report.get("sample_count"):
        raise ValueError("CSVとJSONのサンプル数が一致しません")
    times = np.array([float(row["time_s"]) for row in rows])
    columns = [
        *(f"{name}_qpos" for name in JOINT_NAMES),
        *(f"dice_{axis}_m" for axis in "xyz"),
        *(f"dice_quat_{axis}" for axis in "wxyz"),
    ]
    qpos = np.array([[float(row[key]) for key in columns] for row in rows])
    values = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
    if not all(np.isfinite(a).all() for a in (times, qpos, *values.values())):
        raise ValueError("再生データに非有限値があります")
    step = report.get("timestep_s", 0)
    if (
        not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0
        or not np.allclose(times, np.arange(len(times)) * step, atol=1e-9, rtol=0)
        or not math.isclose(times[-1], report["duration_s"], abs_tol=1e-9)
    ):
        raise ValueError("再生データの時刻が不正です")
    if not np.allclose(np.linalg.norm(qpos[:, 9:13], axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("再生クォータニオンが単位長ではありません")
    if any(row["phase"] not in PHASES for row in rows):
        raise ValueError("未対応の動作区間があります")
    if any(row["top_face"] not in ("", "1", "2", "3", "4", "5", "6") for row in rows):
        raise ValueError("上面番号が不正です")
    return Replay(report, rows, times, qpos, values)


def frame_indices(times: np.ndarray, fps: int) -> np.ndarray:
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("fpsは1〜60の整数にしてください")
    count = max(2, round(float(times[-1] - times[0]) * fps))
    targets = np.linspace(times[0], times[-1], count)
    right = np.searchsorted(times, targets).clip(1, len(times) - 1)
    return np.where(targets - times[right - 1] <= times[right] - targets, right - 1, right)


def draw_metrics(replay: Replay, directory: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=FONT)
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, layout="constrained")
    t, v = replay.times, replay.values
    timing = replay_timing(replay)
    axes[0, 0].plot(t, v["dice_bottom_gap_m"] * 1000)
    axes[0, 0].axhline(0, color="gray", linestyle="--")
    axes[0, 0].set(title="サイコロ下面の天板からの高さ", ylabel="mm")
    faces = [float(row["top_face"]) if row["top_face"] else np.nan for row in replay.rows]
    axes[0, 1].step(t, faces, where="post", label="上面（不確定区間は空白）")
    axes[0, 1].set(title="観測した上面番号", yticks=range(1, 7), ylim=(0.5, 6.5))
    for key, label in (
        ("fixed_jaw_force_n", "固定指"), ("moving_jaw_force_n", "可動指"),
        ("dice_table_force_n", "天板"),
    ):
        axes[1, 0].plot(t, v[key], label=label)
    axes[1, 0].set(title="サイコロの接触法線力", ylabel="N")
    axes[1, 0].legend()
    axes[1, 1].plot(t, v["dice_linear_speed_m_s"], label="線速度 [m/s]")
    axes[1, 1].plot(t, v["dice_angular_speed_rad_s"], label="角速度 [rad/s]")
    axes[1, 1].axhline(.001, color="gray", linestyle="--", label="線速度基準 0.001")
    axes[1, 1].axhline(.01, color="gray", linestyle=":", label="角速度基準 0.01")
    axes[1, 1].set(title="速度（symlog表示）", yscale="symlog", ylim=(0, None))
    axes[1, 1].set_yscale("symlog", linthresh=1e-5)
    axes[1, 1].legend(fontsize=8)
    axes[2, 0].plot(t, v["tilt_deg"])
    axes[2, 0].axhline(5, color="gray", linestyle="--", label="面判定基準 5度")
    axes[2, 0].set(title="最も上向きの面の傾き", ylabel="度")
    axes[2, 0].legend()
    axes[2, 1].plot(t, v["gripper_qpos"])
    axes[2, 1].set(title="グリッパーの実関節角度", ylabel="rad")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.set_xlabel("シミュレーション時刻 [秒]")
        if timing is not None:
            ax.axvspan(timing.support_start_s, timing.support_end_s, alpha=.2, color="orange")
            ax.axvspan(timing.final_start_s, timing.end_s, alpha=.15, color="green")
            for boundary in (
                timing.grasp_end_s, timing.rotation_end_s, timing.rotation_end_s + 4,
                timing.support_end_s, timing.support_end_s + 4, timing.support_end_s + 8,
            ):
                ax.axvline(boundary, color="gray", alpha=.2)
    result = "合格" if replay.report["validation"]["passed"] else "不合格"
    fig.suptitle(
        f"SO-101 保存軌道の検証値（元の物理検証：{result}）\n"
        "配置結果では橙帯＝開指前の支持確認、緑帯＝最終静止評価"
    )
    fig.savefig(directory / "metrics.png", dpi=150)
    fig.savefig(directory / "metrics.svg")
    plt.close(fig)


def add_pips(scene: mujoco.MjvScene, center: np.ndarray, rotation: np.ndarray) -> None:
    # 目は描画シーンだけに追加し、質量・接触形状・保存状態を変えない。
    for number, normal_tuple in FACE_AXES.items():
        normal = np.array(normal_tuple, dtype=float)
        tangent = np.eye(3)[np.argmin(np.abs(normal))]
        bitangent = np.cross(normal, tangent)
        for u, v in PIPS[number]:
            point = normal * .01505 + .007 * (u * tangent + v * bitangent)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .00165),
                center + rotation @ point, np.eye(3).ravel(),
                np.array([.04, .04, .04, 1], dtype=np.float32),
            )
            scene.ngeom += 1


class ReplayRenderer:
    def __init__(self) -> None:
        from PIL import ImageFont

        verify_assets(ROOT / "assets" / "robotstudio_so101")
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
        self.model.vis.global_.offwidth = 640
        self.model.vis.global_.offheight = 480
        self.model.vis.headlight.active = 1
        self.model.vis.headlight.ambient[:] = [.5, .5, .5]
        self.model.vis.headlight.diffuse[:] = [.8, .8, .8]
        self.model.vis.headlight.specular[:] = [.2, .2, .2]
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.option.sitegroup[:] = 0
        self.option.sitegroup[5] = 1
        self.font = ImageFont.truetype(FONT, 21)
        self.small_font = ImageFont.truetype(FONT, 17)
        self.dice = self.model.body("dice").id

    def close(self) -> None:
        self.renderer.close()

    def frame(self, replay: Replay, index: int):
        from PIL import Image, ImageDraw

        self.data.qpos[:] = replay.qpos[index]
        self.data.qvel[:] = 0
        # 再生専用データ。mj_stepは呼ばず、記録姿勢の描画に必要な座標を更新する。
        mujoco.mj_forward(self.model, self.data)
        canvas = Image.new("RGB", (1280, 576), "#17202b")
        for view in range(2):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = (
                [.20, .0, .13] if view == 0 else self.data.xpos[self.dice]
            )
            camera.distance = .90 if view == 0 else .22
            camera.azimuth = 125
            camera.elevation = -28 if view == 0 else -35
            self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
            add_pips(
                self.renderer.scene, self.data.xpos[self.dice],
                self.data.xmat[self.dice].reshape(3, 3),
            )
            canvas.paste(Image.fromarray(self.renderer.render()), (view * 640, 56))
        draw = ImageDraw.Draw(canvas)
        row = replay.rows[index]
        face = row["top_face"] or "不確定"
        phase = phase_label(row["phase"], replay.report.get("requested_face"))
        draw.text(
            (16, 9),
            f"SO-101 {replay.report.get('display_label') or ('観測位置計画' if replay.report.get('planning_mode') == 'observed_position' else '保存軌道再生')}"
            f"   {replay.times[index]:5.2f} 秒   "
            f"{phase}   上面: {face}",
            font=self.font, fill="white",
        )
        draw.text((16, 540), "左：全体　右：サイコロ追従拡大　／　目は描画のみ・物理再計算なし",
                  font=self.small_font, fill="white")
        return canvas


def render_outputs(replay: Replay, directory: Path, fps: int) -> dict:
    from PIL import Image

    indices = frame_indices(replay.times, fps)
    renderer = ReplayRenderer()
    try:
        selected = np.linspace(0, len(replay.times) - 1, 6).round().astype(int)
        timing = replay_timing(replay)
        if timing is not None:
            selected = np.array([
                np.argmin(np.abs(replay.times - time))
                for time in (0, 8, 14, timing.rotation_end_s, timing.support_end_s, timing.end_s)
            ])
        sheet = Image.new("RGB", (1280, 3 * 288), "white")
        for k, index in enumerate(selected):
            image = renderer.frame(replay, int(index))
            sheet.paste(image.resize((640, 288)), ((k % 2) * 640, (k // 2) * 288))
        sheet.save(directory / "contact_sheet.jpg", quality=95)
        renderer.frame(replay, len(replay.times) - 1).save(directory / "final.png")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x576",
            "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(directory / "replay.mp4"),
        ]
        with (directory / "encoder.log").open("xb") as log:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log) as process:
                if process.stdin is None:
                    raise RuntimeError("動画エンコーダーの入力を開けません")
                with process.stdin:
                    for frame, index in enumerate(indices):
                        process.stdin.write(renderer.frame(replay, int(index)).tobytes())
                        if frame % (fps * 5) == 0:
                            print(f"動画描画: {frame}/{len(indices)} フレーム", flush=True)
                if process.wait() != 0:
                    raise RuntimeError("動画エンコード失敗。encoder.logを確認してください")
    finally:
        renderer.close()
    return {
        "fps": fps, "frame_count": len(indices), "video_duration_s": len(indices) / fps,
        "first_sample_index": int(indices[0]), "last_sample_index": int(indices[-1]),
        "sampling": "等間隔の再生時刻に最も近い記録サンプル。両端を含む。補間なし。",
        "resolution": [1280, 576], "backend": "OSMesa CPU", "physics_steps": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="保存したSO-101軌道を動画・グラフへ可視化")
    parser.add_argument("--run-dir", type=Path, required=True, help="result.jsonとtrajectory.csvのフォルダー")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args(argv)
    directory = None
    try:
        replay = load_replay(args.run_dir)
        frame_indices(replay.times, args.fps)
        directory = create_run_directory(args.output_dir)
        draw_metrics(replay, directory)
        video = render_outputs(replay, directory, args.fps)
        write_json(directory / "visualization.json", {
            "source_directory": str(args.run_dir.resolve()),
            "source_sha256": {
                name: hashlib.sha256((args.run_dir / name).read_bytes()).hexdigest()
                for name in ("result.json", "trajectory.csv")
            },
            "source_validation_passed": replay.report["validation"]["passed"],
            "source_task": replay.report["task"], "source_sample_count": len(replay.rows),
            "mujoco_version": mujoco.__version__, **video,
            "notice": "保存状態の描画専用再生。目は非接触の描画要素。物理実行・成功判定の再検証ではない。",
        })
    except (OSError, ValueError, RuntimeError, KeyError, csv.Error) as error:
        print(f"可視化エラー: {type(error).__name__}: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {
                "status": "error", "error_type": type(error).__name__, "message": str(error),
            })
        return 2
    print(f"可視化保存先: {directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
