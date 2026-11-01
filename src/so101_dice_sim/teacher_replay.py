import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from .approach import TIMESTEP_S
from .grasp import capture_sample, validate_grasp
from .learning import FPS, bounded_action, teacher_indices
from .placement import PLACE_GRIPPER_TORQUE_NM, placement_timing, support_ready, validate_placement
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES, ROOT, verify_assets
from .rotation import validate_rotation
from .visualization import Replay, load_replay


MODES = ("source_500hz", "dataset_10hz", "dataset_10hz_bounded")
ENDPOINT_MODES = ("endpoint_linear", "endpoint_linear_bounded", "endpoint_smooth")
ALL_MODES = (*MODES, "source_float32_500hz", *ENDPOINT_MODES)


def interpolate_command(
    start: np.ndarray, endpoint: np.ndarray, step: int, stride: int, *, smooth: bool = False,
) -> np.ndarray:
    if (
        start.shape != (6,) or endpoint.shape != (6,)
        or not np.isfinite(start).all() or not np.isfinite(endpoint).all()
        or stride < 1 or not 1 <= step <= stride
    ):
        raise ValueError("補間の指令形状・値・時間刻みが不正です")
    fraction = step / stride
    if smooth:
        fraction = fraction * fraction * (3 - 2 * fraction)
    if step == stride:
        return endpoint.copy()
    return start + fraction * (endpoint - start)


def action_table(replay: Replay) -> np.ndarray:
    if any(not all(f"{name}_target" in row for name in JOINT_NAMES) for row in replay.rows):
        raise ValueError("教師CSVに関節指令の列がありません")
    actions = np.array([
        [float(row[f"{name}_target"]) for name in JOINT_NAMES] for row in replay.rows
    ])
    if actions.shape != (len(replay.rows), 6) or not np.isfinite(actions).all():
        raise ValueError("教師指令の形状または値が不正です")
    return actions


def held_action_index(physics_step: int, stride: int) -> int:
    if physics_step < 0 or stride < 1:
        raise ValueError("物理ステップ・指令保持間隔が不正です")
    return physics_step // stride


def gate_failures(samples: list[dict], warnings: dict, stage: str) -> list[str]:
    timing = placement_timing(2, observe_position=True)
    if stage == "release":
        return [] if support_ready(samples, target_face=2, observe_position=True) else [
            "supported_before_release",
        ]
    if stage != "rotation_end":
        raise ValueError("未対応の安全ゲートです")
    grasp = validate_grasp(
        samples[:round(timing.grasp_end_s / TIMESTEP_S) + 1], warnings, observe_position=True,
    )
    rotation = validate_rotation(samples, warnings, target_face=2, observe_position=True)
    return [
        *(f"grasp.{key}" for key in grasp["failure_reasons"]),
        *(f"rotation.{key}" for key in rotation["failure_reasons"]),
    ]


def replay_episode(
    directory: Path, source: Path, replay: Replay, dataset_actions: np.ndarray, mode: str,
) -> dict:
    if mode not in ALL_MODES:
        raise ValueError("未対応の再生モードです")
    if (
        replay.report.get("task") != "so101_fixed_place_face_2"
        or replay.report.get("planning_mode") != "observed_position"
        or not replay.report["validation"]["passed"]
        or not np.isclose(replay.report["timestep_s"], TIMESTEP_S, atol=1e-12, rtol=0)
        or len(replay.rows) != round(placement_timing(2, observe_position=True).end_s / TIMESTEP_S) + 1
    ):
        raise ValueError("成功した観測位置計画・面2・500 Hzの教師記録が必要です")
    indices = teacher_indices(len(replay.rows), replay.report["timestep_s"])
    source_actions = action_table(replay)
    if (
        dataset_actions.shape != (len(indices), 6)
        or not np.isfinite(dataset_actions).all()
        or not np.array_equal(dataset_actions, source_actions[np.array(indices) + 1].astype(np.float32))
    ):
        raise ValueError("保存データセットのactionと教師の次ステップ指令が一致しません")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    if (model.nq, model.nv, model.nu, model.neq) != (13, 12, 6, 0):
        raise ValueError("教師再生のモデル構造が一致しません")
    if not np.isclose(model.opt.timestep, TIMESTEP_S, atol=1e-12, rtol=0):
        raise ValueError("教師と物理時間刻みが一致しません")
    if not np.array_equal(
        np.asarray(replay.report["initial_qpos"]), replay.qpos[0],
    ):
        raise ValueError("初期状態と教師CSVが一致しません")
    if replay.report["runtime_overrides"]["gripper_force_range_nm"] != [
        -PLACE_GRIPPER_TORQUE_NM, PLACE_GRIPPER_TORQUE_NM,
    ]:
        raise ValueError("教師の把持トルク制限が一致しません")
    model.actuator_forcerange[5] = [-PLACE_GRIPPER_TORQUE_NM, PLACE_GRIPPER_TORQUE_NM]
    data = mujoco.MjData(model)
    data.qpos[:] = replay.report["initial_qpos"]
    data.ctrl[:] = source_actions[0]
    mujoco.mj_forward(model, data)
    dice = model.body("dice").id
    robot_geoms = [
        i for i in range(model.ngeom)
        if model.geom_bodyid[i] not in (0, dice)
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    for commands in (source_actions, dataset_actions):
        if np.any(commands < model.actuator_ctrlrange[:, 0]) or np.any(
            commands > model.actuator_ctrlrange[:, 1]
        ):
            raise ValueError("教師指令がアクチュエーター範囲外です")
    samples = []
    controls = []
    timing = placement_timing(2, observe_position=True)
    stride = round(1 / FPS / model.opt.timestep)
    endpoint_actions = source_actions[np.array(indices) + stride].astype(np.float32)
    interpolated = mode in ENDPOINT_MODES
    bounded = mode in ("dataset_10hz_bounded", "endpoint_linear_bounded")
    interpolation_start = data.ctrl.copy()
    interpolation_end = data.ctrl.copy()
    rotation_step = round(timing.rotation_end_s / model.opt.timestep)
    release_step = round(timing.support_end_s / model.opt.timestep)
    failures = []
    stopped_before = None
    gates = []
    max_qpos_error = 0.
    pairs = set()

    def warnings() -> dict:
        return {
            mujoco.mjtWarning(i).name: int(w.number)
            for i, w in enumerate(data.warning) if w.number
        }

    def observe(phase: str) -> None:
        nonlocal max_qpos_error
        row, unexpected = capture_sample(
            model, data, phase, robot_geoms=robot_geoms, distance_cap_m=.05,
        )
        pairs.update(unexpected)
        max_qpos_error = max(
            max_qpos_error, float(np.max(np.abs(data.qpos - replay.qpos[len(samples)]))),
        )
        samples.append(row)

    observe("initial")
    for step in range(len(replay.rows) - 1):
        stage = "rotation_end" if step == rotation_step else "release" if step == release_step else None
        if stage is not None:
            failures = gate_failures(samples, warnings(), stage)
            gates.append({"stage": stage, "time_s": float(data.time), "failure_reasons": failures})
            if failures:
                stopped_before = "place_above" if stage == "rotation_end" else "release"
                break
        if mode in ("source_500hz", "source_float32_500hz"):
            raw = source_actions[step + 1]
            data.ctrl[:] = raw.astype(np.float32) if mode == "source_float32_500hz" else raw
        elif step % stride == 0:
            index = held_action_index(step, stride)
            interpolation_start = data.ctrl.copy()
            raw = (endpoint_actions if interpolated else dataset_actions)[index].astype(float)
            applied = (
                bounded_action(raw, data.ctrl, model.actuator_ctrlrange)
                if bounded else raw.copy()
            )
            contact = samples[-1]
            blocked_open = bool(
                bounded
                and (contact["fixed_jaw_force_n"] > .01 or contact["moving_jaw_force_n"] > .01)
                and contact["dice_table_force_n"] <= .01 and applied[5] > data.ctrl[5]
            )
            if blocked_open:
                applied[5] = data.ctrl[5]
            interpolation_end = applied.copy()
            controls.append({
                "time_s": float(data.time), "dataset_frame": index,
                "raw_action_rad": raw.tolist(), "applied_action_rad": applied.tolist(),
                "safety_adjusted": bool(not np.array_equal(raw, applied)),
                "blocked_open": blocked_open,
                **({
                    "start_action_rad": interpolation_start.tolist(),
                    "teacher_endpoint_row": step + stride,
                    "target_time_s": (step + stride) * model.opt.timestep,
                } if interpolated else {}),
            })
            data.ctrl[:] = applied
        if interpolated:
            data.ctrl[:] = interpolate_command(
                interpolation_start, interpolation_end, step % stride + 1, stride,
                smooth=mode == "endpoint_smooth",
            )
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        observe(replay.rows[step + 1]["phase"])
        if bounded and samples[-1]["unexpected_contacts"]:
            failures = ["unexpected_contacts"]
            stopped_before = "next_physics_step"
            break
    completed = len(samples) == len(replay.rows)
    stage_warnings = warnings()
    grasp_validation = (
        validate_grasp(
            samples[:round(timing.grasp_end_s / model.opt.timestep) + 1],
            stage_warnings, observe_position=True,
        ) if len(samples) > round(timing.grasp_end_s / model.opt.timestep) else None
    )
    rotation_validation = (
        validate_rotation(samples[:rotation_step + 1], stage_warnings, target_face=2, observe_position=True)
        if len(samples) > rotation_step else None
    )
    placement_validation = (
        validate_placement(samples, stage_warnings, target_face=2, observe_position=True)
        if completed else None
    )
    if completed:
        failures = [
            *(f"grasp.{key}" for key in grasp_validation["failure_reasons"]),
            *(f"rotation.{key}" for key in rotation_validation["failure_reasons"]),
            *(f"placement.{key}" for key in placement_validation["failure_reasons"]),
        ]
    trajectory = directory / ("trajectory.csv" if completed else "aborted_trajectory.csv")
    with trajectory.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    trajectory_hash = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    identical = trajectory_hash == hashlib.sha256((source / "trajectory.csv").read_bytes()).hexdigest()
    if mode == "source_500hz" and not identical:
        failures.append("source_500hz_trajectory_not_identical")
    report = {
        "mode": mode, "source": str(source.resolve()), "requested_face": 2,
        "action_semantics": (
            "観測tでt+0.1秒末の教師指令を目標とし、前周期の適用末端指令から500Hzで補間。"
            "未来の教師指令を使う上限試験であり、現在の保存データセットのactionとは異なる。"
            if interpolated else "元の教師指令または保存済み次物理ステップ指令の保持。"
        ),
        "passed": completed and not failures, "completed": completed,
        "failure_reasons": failures, "stopped_before_phase": stopped_before,
        "duration_s": float(data.time), "physics_steps": len(samples) - 1,
        "sample_count": len(samples), "control_updates": len(controls),
        "release_executed": any(row["phase"] == "release" for row in samples),
        "controls": controls, "gates": gates,
        "safety_adjusted_count": sum(row["safety_adjusted"] for row in controls),
        "blocked_open_count": sum(row["blocked_open"] for row in controls),
        "grasp_validation": grasp_validation, "rotation_validation": rotation_validation,
        "placement_validation": placement_validation, "warnings": stage_warnings,
        "unexpected_contact_pairs": [list(pair) for pair in sorted(pairs)],
        "final_state": samples[-1], "initial_qpos": replay.report["initial_qpos"],
        "max_qpos_deviation_from_source": max_qpos_error,
        "trajectory_sha256": trajectory_hash, "source_trajectory_identical": identical,
        "notice": (
            "指令だけで物理再生。物体状態の代入は初期化だけ。保存時刻の安全ゲートを維持。"
            "生指令モードも既存の把持/回転/支持ゲートで中断し、安全制限モードは毎物理ステップで想定外接触も中断。"
        ),
    }
    write_json(directory / "replay.json", report)
    return report


def run_study(
    directory: Path, collection_path: Path, modes: tuple[str, ...] = MODES,
) -> dict:
    if not modes or len(set(modes)) != len(modes) or any(mode not in ALL_MODES for mode in modes):
        raise ValueError("比較モードが空、重複、または未対応です")
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    verify_assets(ROOT / "assets" / "robotstudio_so101")
    dataset_root = Path(collection["dataset"])
    dataset = LeRobotDataset("local/so101_face2_smoke", root=dataset_root)
    if collection["fps"] != FPS or dataset.fps != FPS or collection["joint_units"] != "rad":
        raise ValueError("データセットの周期または関節単位が一致しません")
    if len(collection["episodes"]) != dataset.meta.total_episodes:
        raise ValueError("収集記録とエピソード数が一致しません")
    manifest = {
        "collection": str(collection_path.resolve()),
        "collection_sha256": hashlib.sha256(collection_path.read_bytes()).hexdigest(),
        "dataset_sha256": {
            str(path.relative_to(dataset_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(dataset_root.rglob("*.parquet"))
        },
        "modes": modes, "uploaded": False,
        "endpoint_target_offset_s": 1 / FPS,
    }
    write_json(directory / "settings.json", manifest)
    columns = dataset.hf_dataset.select_columns(["action", "episode_index", "frame_index"])
    offset = 0
    results = []
    for episode_index, episode in enumerate(collection["episodes"]):
        source = Path(episode["source"])
        for name, digest in episode["source_sha256"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"元教師記録のハッシュが一致しません: {source / name}")
        replay = load_replay(source)
        frames = episode["frames"]
        values = columns[offset:offset + frames]
        if not np.array_equal(np.asarray(values["episode_index"]).reshape(-1), np.full(frames, episode_index)):
            raise ValueError("データセットのエピソード境界が一致しません")
        if not np.array_equal(np.asarray(values["frame_index"]).reshape(-1), np.arange(frames)):
            raise ValueError("データセットのフレーム順が一致しません")
        actions = np.asarray(values["action"], dtype=np.float32)
        for mode in modes:
            run = directory / f"episode-{episode_index:02d}-{mode}"
            run.mkdir()
            report = replay_episode(run, source, replay, actions, mode)
            results.append({
                "episode": episode_index, "mode": mode, "run_directory": run.name,
                **{key: report[key] for key in (
                    "passed", "completed", "duration_s", "failure_reasons",
                    "stopped_before_phase", "control_updates", "safety_adjusted_count",
                    "source_trajectory_identical",
                )},
            })
            print(
                f"教師再生 {episode_index}/{mode}: {'成功' if report['passed'] else '不合格'} "
                f"{report['failure_reasons']}", flush=True,
            )
        offset += frames
    if offset != len(dataset):
        raise ValueError("収集記録と全フレーム数が一致しません")
    summary = {
        "results": results, "runs": len(results), "passed_count": sum(row["passed"] for row in results),
        "modes": {
            mode: {"passed": sum(row["passed"] for row in results if row["mode"] == mode),
                   "runs": sum(row["mode"] == mode for row in results)}
            for mode in modes
        },
        "uploaded": False,
        "notice": "既存3初期条件だけの教師再生。学習済み方策の成功や未知条件への汎化ではない。",
    }
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="保存した面2教師指令の500 Hz/10 Hz物理再生を比較")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--modes", choices=ALL_MODES, nargs="+", default=MODES)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/teacher-replays"))
    args = parser.parse_args(argv)
    directory = None
    try:
        directory = create_run_directory(args.output_dir)
        report = run_study(directory, args.collection, tuple(args.modes))
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"教師再生エラー: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {"status": "error", "message": str(error)})
        return 2
    print(f"教師再生結果: {directory.resolve()}", flush=True)
    return 0 if report["passed_count"] == report["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
