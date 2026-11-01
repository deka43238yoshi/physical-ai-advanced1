import argparse
from pathlib import Path
import sys

from .batch import run_batch
from .reporting import create_run_directory, save_result, write_json
from .settings import load_settings
from .simulation import run_simulation, validate_result
from .robot_check import run_robot_check
from .approach import run_approach
from .grasp import run_grasp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="サイコロのヘッドレス物理検証")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--check-place", action="store_true",
        help="指定面で接地・解放・退避・静止を検証（既定は固定条件・面2）",
    )
    parser.add_argument(
        "--target-face", type=int, choices=range(1, 7), metavar="N",
        help="--check-placeの目標面。対応は1・2・3、面4〜6は調査段階",
    )
    parser.add_argument(
        "--observe-position", action="store_true",
        help="着地後のMuJoCo観測位置から把持目標を計画（--check-place専用）",
    )
    parser.add_argument(
        "--initial-offset-mm", type=float, nargs=2, metavar=("DX", "DY"),
        help="観測計画の初期XYずれ、各±5 mm以内（姿勢固定）",
    )
    parser.add_argument(
        "--check-rotate", action="store_true",
        help="固定把持後に90度回転し、上面2と2秒保持を検証（配置なし）",
    )
    parser.add_argument(
        "--check-grasp", action="store_true",
        help="固定条件で下降・閉指・持ち上げ・2秒保持を検証",
    )
    parser.add_argument(
        "--check-approach", action="store_true",
        help="SO-101・テーブル・サイコロの統合シーンで接触前の到達を検証",
    )
    parser.add_argument(
        "--check-so101", action="store_true",
        help="SO-101単体の関節駆動・保持・開閉を検証",
    )
    parser.add_argument(
        "--random-position-runs", type=int, metavar="N",
        help="位置を指定範囲で一様抽選してN回実行",
    )
    parser.add_argument(
        "--random-orientation", action="store_true",
        help="バッチ実行で初期姿勢も全方向から一様抽選",
    )
    parser.add_argument("--seed", type=int, default=42, help="抽選のシード（既定42）")
    args = parser.parse_args(argv)
    if args.observe_position and not args.check_place:
        parser.error("--observe-positionには--check-placeが必要です")
    if args.initial_offset_mm is not None and not args.observe_position:
        parser.error("--initial-offset-mmには--observe-positionが必要です")
    if args.target_face is not None and not args.check_place:
        parser.error("--target-faceには--check-placeが必要です")
    if args.random_orientation and args.random_position_runs is None:
        parser.error("--random-orientationには--random-position-runsが必要です")
    if args.check_so101 and args.random_position_runs is not None:
        parser.error("--check-so101とサイコロのランダム実行は併用できません")
    if args.check_approach and (args.check_so101 or args.random_position_runs is not None):
        parser.error("--check-approachは他の検証モードと併用できません")
    if args.check_grasp and (
        args.check_so101 or args.check_approach or args.random_position_runs is not None
    ):
        parser.error("--check-graspは他の検証モードと併用できません")
    if args.check_rotate and (
        args.check_grasp or args.check_so101 or args.check_approach
        or args.random_position_runs is not None
    ):
        parser.error("--check-rotateは他の検証モードと併用できません")
    if args.check_place and (
        args.check_grasp or args.check_so101 or args.check_approach or args.check_rotate
        or args.random_position_runs is not None
    ):
        parser.error("--check-placeは他の検証モードと併用できません")
    directory = None
    try:
        directory = create_run_directory(args.output_dir)
        if args.check_place:
            face = 2 if args.target_face is None else args.target_face
            report = run_grasp(
                directory, place_face=face, observe_position=args.observe_position,
                initial_offset_m=(
                    (0.0, 0.0) if args.initial_offset_mm is None
                    else tuple(value / 1000 for value in args.initial_offset_mm)
                ),
            )
            print(f"結果保存先: {directory.resolve()}")
            if not report["validation"]["passed"]:
                print(
                    "配置検証失敗: " + ", ".join(report["validation"]["failure_reasons"]),
                    file=sys.stderr,
                )
                return 1
            mode = "観測位置からの計画" if args.observe_position else "固定把持"
            print(f"配置検証成功: {mode}から上面{face}で接地・解放・退避後静止")
            return 0
        if args.check_rotate:
            report = run_grasp(directory, rotate_to_face=2)
            print(f"結果保存先: {directory.resolve()}")
            if not report["validation"]["passed"]:
                print(
                    "回転検証失敗: " + ", ".join(report["validation"]["failure_reasons"]),
                    file=sys.stderr,
                )
                return 1
            print("回転検証成功: 固定把持後に上面2へ回転・2秒保持（配置なし）")
            return 0
        if args.check_grasp:
            report = run_grasp(directory)
            print(f"結果保存先: {directory.resolve()}")
            if not report["validation"]["passed"]:
                print(
                    "把持検証失敗: " + ", ".join(report["validation"]["failure_reasons"]),
                    file=sys.stderr,
                )
                return 1
            print("把持検証成功: 両指接触・30 mm以上の離床・2秒保持（配置は未実施）")
            return 0
        if args.check_approach:
            report = run_approach(directory)
            print(f"結果保存先: {directory.resolve()}")
            if not report["validation"]["passed"]:
                print(
                    "接近検証失敗: " + ", ".join(report["validation"]["failure_reasons"]),
                    file=sys.stderr,
                )
                return 1
            print("接近検証成功: サイコロ上方へ到達、把持は未実施")
            return 0
        if args.check_so101:
            report = run_robot_check(directory)
            print(f"結果保存先: {directory.resolve()}")
            if not report["validation"]["passed"]:
                print(
                    "SO-101検証失敗: "
                    + ", ".join(report["validation"]["failure_reasons"]),
                    file=sys.stderr,
                )
                return 1
            print("SO-101単体検証成功: 6関節、保持・小移動・開閉、接触なし")
            return 0
        settings = load_settings(args.config)
        if args.random_position_runs is not None:
            summary = run_batch(
                directory, settings, args.random_position_runs, args.seed,
                randomize_orientation=args.random_orientation,
            )
            print(f"結果保存先: {directory.resolve()}")
            print(f"合格: {summary['passed_count']}/{summary['runs']}")
            if not summary["passed"]:
                print("物理検証失敗: summary.jsonを確認してください", file=sys.stderr)
                return 1
            return 0
        result = run_simulation(settings)
        validation = validate_result(result, settings)
        save_result(directory, settings, result, validation)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"実行エラー: {type(error).__name__}: {error}", file=sys.stderr)
        if directory is not None:
            try:
                write_json(directory / "error.json", {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                })
            except (OSError, ValueError) as save_error:
                print(f"診断結果の保存にも失敗: {save_error}", file=sys.stderr)
        return 2
    print(f"結果保存先: {directory.resolve()}")
    if not validation["passed"]:
        print(
            "物理検証失敗: " + ", ".join(validation["failure_reasons"]),
            file=sys.stderr,
        )
        return 1
    print(
        f"物理検証成功: 上面={result.samples[-1].top_face}, "
        f"最終1秒の最大線速度={validation['max_linear_speed_m_s']:.6g} m/s, "
        f"最大角速度={validation['max_angular_speed_rad_s']:.6g} rad/s"
    )
    return 0
