import argparse
import csv
from pathlib import Path
import random
import sys

from .grasp import run_grasp
from .observation import POSITION_OFFSET_LIMIT_M, validate_position_offset
from .reporting import create_run_directory, write_json


def position_cases(seed: int, random_samples: int) -> list[tuple[float, float]]:
    if type(random_samples) is not int or random_samples < 0:
        raise ValueError("追加ランダム位置数は0以上の整数にしてください")
    limit = POSITION_OFFSET_LIMIT_M
    generator = random.Random(seed)
    return [
        (x, y) for x in (-limit, 0.0, limit) for y in (-limit, 0.0, limit)
    ] + [
        (generator.uniform(-limit, limit), generator.uniform(-limit, limit))
        for _ in range(random_samples)
    ]


def run_position_batch(
    directory: Path, offsets: list[tuple[float, float]], *, seed: int,
    grasp_error_m: tuple[float, float] = (0.0, 0.0),
) -> dict:
    if not offsets:
        raise ValueError("検証する位置がありません")
    positions = [validate_position_offset(offset).tolist() for offset in offsets]
    injected_error = validate_position_offset(grasp_error_m).tolist()
    write_json(directory / "batch_settings.json", {
        "seed": seed, "initial_xy_offsets_m": positions, "target_faces": [1, 2, 3],
        "offset_limit_m": POSITION_OFFSET_LIMIT_M,
        "injected_grasp_error_m": injected_error,
        "initial_orientation": "上面1・方位固定",
        "policy": "同じ位置ケースを全3面で実行。失敗も保存して残りを継続。",
    })
    results = []
    for index, position in enumerate(positions, 1):
        for face in (1, 2, 3):
            run = directory / f"position-{index:02d}-face-{face}"
            run.mkdir(exist_ok=False)
            row = {
                "position_case": index, "target_face": face,
                "offset_x_m": position[0], "offset_y_m": position[1],
                "run_directory": run.name,
            }
            try:
                report = run_grasp(
                    run, place_face=face, observe_position=True,
                    initial_offset_m=tuple(position),
                    grasp_error_m=tuple(injected_error),
                )
                row.update({
                    "status": "passed" if report["validation"]["passed"] else "validation_failed",
                    "passed": report["validation"]["passed"],
                    "failure_reason": ", ".join(report["validation"]["failure_reasons"]),
                    "final_top_face": report["final_state"]["top_face"],
                    "min_robot_clearance_m": report["validation"]["min_robot_clearance_m"],
                })
            except (ValueError, RuntimeError, FloatingPointError) as error:
                write_json(run / "error.json", {
                    "status": "error", "error_type": type(error).__name__, "message": str(error),
                })
                row.update({
                    "status": "execution_error", "passed": False, "failure_reason": str(error),
                    "final_top_face": None, "min_robot_clearance_m": None,
                })
                print(f"位置{index} 面{face} 実行エラー: {error}", file=sys.stderr, flush=True)
            results.append(row)
            print(f"位置{index}/{len(positions)} 面{face}: {row['status']}", flush=True)
    summary = {
        "runs": len(results), "passed_count": sum(r["passed"] for r in results),
        "passed": all(r["passed"] for r in results), "seed": seed, "results": results,
        "notice": "有限個の検証結果。±5 mmの連続領域全体の成功保証ではない。",
    }
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="観測位置計画の境界・中心と乱数位置を全3面で検証")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/position-batches"))
    args = parser.parse_args(argv)
    directory = None
    try:
        offsets = position_cases(args.seed, args.random_samples)
        directory = create_run_directory(args.output_dir)
        summary = run_position_batch(directory, offsets, seed=args.seed)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"位置バッチエラー: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {"status": "error", "message": str(error)})
        return 2
    print(f"位置バッチ保存先: {directory.resolve()}", flush=True)
    print(f"合格: {summary['passed_count']}/{summary['runs']}", flush=True)
    if any(r["status"] == "execution_error" for r in summary["results"]):
        return 2
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
