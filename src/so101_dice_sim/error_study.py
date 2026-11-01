import argparse
import csv
import math
from pathlib import Path
import sys

from .position_batch import run_position_batch
from .reporting import create_run_directory, write_json


def error_cases(radius_mm: float) -> list[tuple[float, float]]:
    if not math.isfinite(radius_mm) or not 0 < radius_mm <= 5:
        raise ValueError("誤差半径は0より大きく5 mm以下にしてください")
    radius = radius_mm / 1000
    return [(0., 0.)] + [
        (radius * math.cos(angle * math.pi / 4), radius * math.sin(angle * math.pi / 4))
        for angle in range(8)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="人工的な把持XY誤差の配置成功への影響を測定")
    parser.add_argument("--radius-mm", type=float, default=3.1)
    parser.add_argument("--initial-offset-mm", nargs=2, type=float, default=(0., 0.))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grasp-error-studies"))
    args = parser.parse_args(argv)
    directory = None
    try:
        errors = error_cases(args.radius_mm)
        directory = create_run_directory(args.output_dir)
        write_json(directory / "study_settings.json", {
            "radius_mm": args.radius_mm, "grasp_errors_m": errors,
            "initial_offset_mm": args.initial_offset_mm, "target_faces": [1, 2, 3],
            "policy": "初期位置は変えず観測後の把持XYだけに誤差注入。基準0と8方向。閾値変更なし。",
        })
        results = []
        for index, error in enumerate(errors):
            run = directory / f"error-{index:02d}"
            run.mkdir()
            summary = run_position_batch(
                run, [tuple(value / 1000 for value in args.initial_offset_mm)],
                seed=42, grasp_error_m=error,
            )
            for row in summary["results"]:
                results.append({
                    "error_case": index, "grasp_error_x_mm": error[0] * 1000,
                    "grasp_error_y_mm": error[1] * 1000,
                    **row, "run_directory": f"{run.name}/{row['run_directory']}",
                })
            print(f"誤差条件{index}: {summary['passed_count']}/3", flush=True)
        report = {
            "runs": len(results), "passed_count": sum(row["passed"] for row in results),
            "results": results,
            "notice": "固定初期位置・姿勢と有限8方向での誤差耐性測定。全位置・連続誤差の保証ではない。",
        }
        write_json(directory / "summary.json", report)
        with (directory / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"誤差耐性検証エラー: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {"status": "error", "message": str(error)})
        return 2
    print(f"誤差耐性結果: {directory.resolve()}", flush=True)
    print(f"成功: {report['passed_count']}/{report['runs']}", flush=True)
    return 0 if report["passed_count"] == report["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
