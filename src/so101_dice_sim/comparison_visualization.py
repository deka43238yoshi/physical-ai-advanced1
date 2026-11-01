import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .endpoint_collection import load_endpoint_replay
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES
from .visualization import (
    FONT, METRICS, PHASES, Replay, ReplayRenderer, frame_indices, render_outputs,
)


def load_comparison(success: Path, policy_path: Path) -> tuple[Replay, Replay]:
    teacher, _ = load_endpoint_replay(success)
    report = json.loads((policy_path / "evaluation.json").read_text(encoding="utf-8"))
    settings_path = policy_path.parent / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    split = settings["split"]
    allowed_offsets = (
        split["train_offsets_mm"] if settings.get("evaluation_type") == "training_position_diagnostic"
        else split["evaluation_offsets_mm"]
    )
    target_face = settings.get("target_face", split["target_face"])
    if (
        type(report["passed"]) is not bool or type(report["completed"]) is not bool
        or report["requested_face"] != target_face or target_face != 2
        or report["offset_mm"] not in allowed_offsets
        or (report["passed"] and not report["completed"])
        or (not report["passed"] and not report["failure_reasons"])
    ):
        raise ValueError("面2・所定位置の完了または中断した方策評価が必要です")
    with (policy_path / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row["time_s"]) for row in rows])
    columns = [
        *(f"{name}_qpos" for name in JOINT_NAMES),
        *(f"dice_{axis}_m" for axis in "xyz"),
        *(f"dice_quat_{axis}" for axis in "wxyz"),
    ]
    metrics = (*METRICS, "unexpected_contacts")
    required = {"time_s", "phase", "top_face", *columns, *metrics}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("再生に必要なCSV列がありません")
    qpos = np.array([[float(row[key]) for key in columns] for row in rows])
    values = {key: np.array([float(row[key]) for row in rows]) for key in metrics}
    if (
        len(rows) < 2 or len(rows) != report["physics_steps"] + 1
        or not all(np.isfinite(value).all() for value in (times, qpos, *values.values()))
        or not np.allclose(times, np.arange(len(times)) * .002, atol=1e-9, rtol=0)
        or not np.isclose(times[-1], report["duration_s"], atol=1e-9, rtol=0)
        or not np.allclose(np.linalg.norm(qpos[:, 9:13], axis=1), 1, atol=1e-8, rtol=0)
        or not np.allclose(qpos[0], teacher.qpos[0], atol=1e-12, rtol=0)
        or times[-1] > teacher.times[-1]
        or any(row["phase"] not in PHASES for row in rows)
        or any(row["top_face"] not in ("", "1", "2", "3", "4", "5", "6") for row in rows)
    ):
        raise ValueError("比較軌道の時刻・状態・初期条件が整合しません")
    if report["failure_reasons"] and any(
        reason.startswith("unexpected_contacts:") for reason in report["failure_reasons"]
    ) and (
        values["unexpected_contacts"][-1] <= 0
        or np.any(values["unexpected_contacts"][:-1] != 0)
    ):
        raise ValueError("想定外接触による停止位置とCSVが一致しません")
    if report["passed"] and np.any(values["unexpected_contacts"] != 0):
        raise ValueError("成功判定軌道に想定外接触があります")
    for key in ("time_s", *columns, *metrics):
        if not np.isclose(float(rows[-1][key]), report["final_state"][key], atol=1e-12, rtol=0):
            raise ValueError("CSVと中断時の最終状態が一致しません")
    teacher.report = {
        **teacher.report, "task": "so101_fixed_place_face_2",
        "planning_mode": "observed_position", "validation": {"passed": True},
        "display_label": "教師制御（最終結果：面2配置成功）",
    }
    teacher.values = {key: np.array([float(row[key]) for row in teacher.rows]) for key in metrics}
    policy_replay = Replay(
        {**report, "task": "smolvla_evaluation", "requested_face": 2,
         "validation": {"passed": report["passed"]},
         "evaluation_settings": str(settings_path.resolve())},
        rows, times, qpos, values,
    )
    policy_replay.report["display_label"] = (
        "SmolVLA（最終結果：面2配置成功）" if report["passed"]
        else "SmolVLA（最終結果：安全中断）"
    )
    return teacher, policy_replay


def comparison_frame(renderer, teacher, policy_replay, index):
    from PIL import Image, ImageDraw, ImageFont

    time_s = policy_replay.times[index]
    teacher_index = int(np.argmin(np.abs(teacher.times - time_s)))
    canvas = Image.new("RGB", (1280, 400), "#17202b")
    for column, replay, sample in ((0, teacher, teacher_index), (1, policy_replay, index)):
        canvas.paste(renderer.frame(replay, sample).resize((640, 288)), (column * 640, 50))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 21)
    draw.text((12, 10), "成功：教師制御（SmolVLAではありません）", font=font, fill="#7ee787")
    right_title = (
        "成功：SmolVLA（学習済み方策）" if policy_replay.report["passed"]
        else "失敗：SmolVLA（学習済み方策）"
    )
    draw.text((652, 10), right_title, font=font, fill="#7ee787" if policy_replay.report["passed"] else "#ff9999")
    draw.text((12, 342), f"同一初期位置・同一シミュレーション時刻 {time_s:.3f} 秒", font=font, fill="white")
    message = "右：方策の最終記録状態。" if (
        index == len(policy_replay.times) - 1 and policy_replay.report["passed"]
    ) else (
        "右：方策が安全停止。この先の動作はありません。"
        if index == len(policy_replay.times) - 1
        else "両列とも保存状態の描画のみ。物理の再実行・状態補間なし。"
    )
    draw.text((12, 370), message, font=font, fill="white")
    return canvas


def draw_comparison(teacher, policy_replay, directory):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    plt.rcParams["font.family"] = FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
    metrics = [
        ("dice_bottom_gap_m", "サイコロ下面の高さ [m]"),
        ("unexpected_contacts", "想定外接触数（右の末端で天板に接触）"),
        ("fixed_jaw_force_n", "固定指の接触力 [N]"),
        ("dice_table_force_n", "天板の支持力 [N]"),
    ]
    for axis, (key, title) in zip(axes.flat, metrics):
        policy_label = "SmolVLA（成功）" if policy_replay.report["passed"] else "SmolVLA（失敗）"
        for replay, label in ((teacher, "成功教師"), (policy_replay, policy_label)):
            mask = replay.times <= policy_replay.times[-1] + 1e-9
            axis.plot(replay.times[mask], replay.values[key][mask], label=label)
        axis.set(title=title, xlabel="シミュレーション時刻 [s]")
        axis.legend()
    policy_label = "SmolVLA（成功）" if policy_replay.report["passed"] else "SmolVLA（失敗）"
    for replay, label in ((teacher, "成功教師"), (policy_replay, policy_label)):
        mask = replay.times <= policy_replay.times[-1] + 1e-9
        for column, joint in enumerate(("shoulder_lift", "wrist_flex")):
            values = np.array([float(row[f"{joint}_qpos"]) for row in replay.rows])
            axes[2, column].plot(replay.times[mask], values[mask], label=label)
            axes[2, column].set(title=f"{joint} 実関節角 [rad]", xlabel="シミュレーション時刻 [s]")
            axes[2, column].legend()
    if not policy_replay.report["passed"]:
        for axis in axes.flat:
            axis.axvline(policy_replay.times[-1], color="red", linestyle="--", label="安全停止")
    outcome = "SmolVLAも面2配置に成功" if policy_replay.report["passed"] else "SmolVLAは安全停止"
    fig.suptitle(
        f"同一初期位置の比較：教師は45秒の配置成功／{outcome}\n"
        f"比較範囲は方策記録の最終時刻 {policy_replay.times[-1]:.3f} 秒まで。停止後の値を延長しない。"
    )
    fig.savefig(directory / "comparison_metrics.png", dpi=150)
    fig.savefig(directory / "comparison_metrics.svg")
    plt.close(fig)


def render_comparison(directory, success, policy_path, fps):
    from PIL import Image

    teacher, policy_replay = load_comparison(success, policy_path)
    indices = frame_indices(policy_replay.times, fps)
    for name, replay in (("success_teacher", teacher), ("policy_smolvla", policy_replay)):
        output = directory / name
        output.mkdir()
        write_json(output / "visualization.json", {
            **render_outputs(replay, output, fps),
            "controller": name, "passed": replay.report["validation"]["passed"],
        })
    renderer = ReplayRenderer()
    try:
        sheet = Image.new("RGB", (1280, 1200))
        for row, index in enumerate((
            0, int(np.argmin(np.abs(policy_replay.times - min(5, policy_replay.times[-1])))),
            len(policy_replay.times) - 1,
        )):
            sheet.paste(comparison_frame(renderer, teacher, policy_replay, index), (0, row * 400))
        sheet.save(directory / "comparison_sheet.jpg", quality=95)
        last_image = "final_comparison.png" if policy_replay.report["passed"] else "failure_instant.png"
        comparison_frame(renderer, teacher, policy_replay, len(policy_replay.times) - 1).save(
            directory / last_image,
        )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "1280x400", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "comparison.mp4"),
        ]
        with (directory / "comparison_encoder.log").open("xb") as log:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log) as process:
                if process.stdin is None:
                    raise RuntimeError("動画入力を開けません")
                with process.stdin:
                    for index in indices:
                        process.stdin.write(
                            comparison_frame(renderer, teacher, policy_replay, int(index)).tobytes(),
                        )
                if process.wait() != 0:
                    raise RuntimeError("比較動画エンコード失敗")
    finally:
        renderer.close()
    draw_comparison(teacher, policy_replay, directory)
    report = {
        "success_type": "教師制御・安全制限付き線形補間", "policy_type": "学習済みSmolVLA",
        "success": str(success.resolve()), "policy": str(policy_path.resolve()),
        "policy_passed": policy_replay.report["passed"],
        "source_sha256": {
            label: {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names}
            for label, path, names in (
                ("success", success, ("replay.json", "trajectory.csv")),
                ("policy", policy_path, ("evaluation.json", "trajectory.csv")),
            )
        },
        "offset_mm": policy_replay.report["offset_mm"],
        "failure_reasons": policy_replay.report["failure_reasons"],
        "evaluation_settings": policy_replay.report["evaluation_settings"],
        "settings_sha256": hashlib.sha256(Path(policy_replay.report["evaluation_settings"]).read_bytes()).hexdigest(),
        "success_duration_s": float(teacher.times[-1]), "policy_duration_s": float(policy_replay.times[-1]),
        "comparison_frame_count": len(indices), "fps": fps, "physics_steps": 0,
        "notice": "教師とSmolVLAの成功状態を個別に記録。比較は方策記録の最終時点まで。各列左は全体、右は物体追従拡大。",
    }
    write_json(directory / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="成功教師と失敗SmolVLAを同一初期状態・時刻で可視化")
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument(
        "--policy", "--failure", dest="policy", type=Path, required=True,
        help="方策評価のevaluation.jsonがあるフォルダー",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/comparisons"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        render_comparison(directory, args.success, args.policy, args.fps)
    except (OSError, ValueError, RuntimeError, KeyError, csv.Error) as error:
        write_json(directory / "error.json", {"message": str(error)})
        print(str(error), file=sys.stderr)
        return 2
    print(f"可視化保存先: {directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
