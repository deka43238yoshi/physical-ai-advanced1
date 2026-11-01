import csv
from dataclasses import asdict, replace
import math
from pathlib import Path
import random

from .reporting import save_result, write_json
from .settings import Settings
from .simulation import run_simulation, validate_result


POSITION_BOUNDS_M = ((-0.10, 0.10), (-0.10, 0.10), (0.46, 0.55))


def random_positions(count: int, seed: int) -> list[tuple[float, float, float]]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("ランダム位置での実行回数は正の整数にしてください")
    generator = random.Random(seed)
    return [
        tuple(generator.uniform(low, high) for low, high in POSITION_BOUNDS_M)
        for _ in range(count)
    ]


def random_orientations(
    count: int, seed: int
) -> list[tuple[float, float, float, float]]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("ランダム姿勢での実行回数は正の整数にしてください")
    # 位置の抽選列を維持する独立ストリーム。S^3上の一様分布をwxyz順にする。
    generator = random.Random(f"orientation:{seed}")
    quaternions = []
    for _ in range(count):
        u1, u2, u3 = (generator.random() for _ in range(3))
        quaternions.append((
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
        ))
    return quaternions


def run_batch(
    directory: Path, settings: Settings, count: int, seed: int,
    randomize_orientation: bool = False,
) -> dict:
    positions = random_positions(count, seed)
    orientations = (
        random_orientations(count, seed) if randomize_orientation
        else [settings.initial_quaternion_wxyz] * count
    )
    # 実行途中でエラーになっても抽選済み条件は残す。
    write_json(directory / "batch_settings.json", {
        "seed": seed,
        "distribution": "independent_uniform_xyz",
        "position_bounds_m": dict(zip(("x", "y", "z"), POSITION_BOUNDS_M)),
        "base_settings": asdict(settings),
        "initial_positions_m": positions,
        "orientation_distribution": (
            "uniform_so3" if randomize_orientation else "fixed"
        ),
        "initial_quaternions_wxyz": orientations,
    })
    rows = []
    for index, (position, orientation) in enumerate(
        zip(positions, orientations), start=1
    ):
        run_settings = replace(
            settings, initial_position_m=position,
            initial_quaternion_wxyz=orientation,
        )
        run_directory = directory / f"run-{index:02d}"
        run_directory.mkdir(exist_ok=False)
        result = run_simulation(run_settings)
        validation = validate_result(result, run_settings)
        save_result(run_directory, run_settings, result, validation)
        row = {
            "run": index,
            "initial_x_m": position[0],
            "initial_y_m": position[1],
            "initial_z_m": position[2],
            **{
                f"initial_quaternion_{axis}": value
                for axis, value in zip("wxyz", orientation)
            },
            "passed": validation["passed"],
            "top_face": result.samples[-1].top_face,
            "tilt_deg": result.samples[-1].tilt_deg,
            "max_linear_speed_m_s": validation["max_linear_speed_m_s"],
            "max_angular_speed_rad_s": validation["max_angular_speed_rad_s"],
            "max_abs_bottom_gap_m": validation["max_abs_bottom_gap_m"],
            "failure_reasons": ",".join(validation["failure_reasons"]),
            "result_relative_path": f"run-{index:02d}/result.json",
        }
        rows.append(row)
        print(
            f"実行{index}/{count}: 初期位置={position}, "
            f"合格={validation['passed']}, 上面={row['top_face']}"
        )
    summary = {
        "seed": seed,
        "randomize_orientation": randomize_orientation,
        "runs": count,
        "passed_count": sum(row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
        "results": rows,
    }
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary
