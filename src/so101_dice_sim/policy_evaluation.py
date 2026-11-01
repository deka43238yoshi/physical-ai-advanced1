import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from .action_contract import ENDPOINT_MODE, require_training_contract
from .grasp import capture_sample, validate_grasp
from .learning import TASK, bounded_action, load_policy, render_observation, training_normalization_stats
from .learning_study import split_positions
from .placement import PLACE_INITIAL_POSITION_M, stable_support_window, validate_placement
from .reporting import create_run_directory, write_json
from .rotation import validate_rotation
from .teacher_replay import gate_failures, interpolate_command
from .visualization import ReplayRenderer


def evaluation_phase(step: int) -> str:
    if step == 0:
        return "initial"
    t = step * .002
    if t >= 44:
        return "final_evaluation"
    if 23 <= t <= 25:
        return "rotation_hold"
    if 17 <= t <= 19:
        return "hold"
    return "policy"


def require_initial_qpos(actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if (
        actual.shape != (13,) or expected.shape != (13,)
        or not np.isfinite(actual).all() or not np.isfinite(expected).all()
        or not np.array_equal(actual, expected)
    ):
        raise ValueError("評価環境の初期qposが成功教師と完全一致しません")


def evaluate_position(
    directory: Path, policy, pre, post, offset_mm, seed: int, *,
    expected_initial_qpos: np.ndarray | None = None,
    evaluation_type: str = "held_out_position",
) -> dict:
    import torch

    torch.manual_seed(seed)
    policy.eval()
    policy.reset()
    renderer = ReplayRenderer()
    model = renderer.model
    if (model.nq, model.nv, model.nu, model.neq) != (13, 12, 6, 0) or not np.isclose(
        model.opt.timestep, .002, atol=1e-12, rtol=0,
    ):
        renderer.close()
        raise ValueError("評価用モデルの構造または物理時間刻みが契約と一致しません")
    model.actuator_forcerange[5] = [-.25, .25]
    data = mujoco.MjData(model)
    data.qpos[5] = .6
    data.qpos[6:9] = PLACE_INITIAL_POSITION_M
    data.qpos[6:8] += np.asarray(offset_mm) / 1000
    if expected_initial_qpos is not None:
        expected_initial_qpos = np.asarray(expected_initial_qpos, dtype=float)
        try:
            require_initial_qpos(data.qpos, expected_initial_qpos)
        except ValueError:
            renderer.close()
            raise
    initial_qpos = data.qpos.copy()
    data.ctrl[:] = data.qpos[:6]
    mujoco.mj_forward(model, data)
    dice = model.body("dice").id
    robot_geoms = [
        i for i in range(model.ngeom) if model.geom_bodyid[i] not in (0, dice)
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    samples = []
    controls = []
    reason = []
    grasp = rotation = placement = None
    try:
        row, _ = capture_sample(model, data, "initial", robot_geoms=robot_geoms, distance_cap_m=.05)
        if expected_initial_qpos is not None and row["top_face"] != 1:
            raise ValueError("診断評価の初期上面が教師条件の面1と一致しません")
        samples.append(row)
        for cycle in range(450):
            if cycle == 250:
                reason = gate_failures(samples, {}, "rotation_end")
                if reason:
                    break
            if cycle == 340 and not stable_support_window(samples[-251:], target_face=2):
                reason = ["supported_before_release"]
                break
            observation = {
                key: torch.from_numpy(image).permute(2, 0, 1).float() / 255
                for key, image in render_observation(renderer, data.qpos).items()
            }
            observation.update({"observation.state": torch.tensor(data.qpos[:6], dtype=torch.float32),
                                "task": TASK})
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = post(policy.select_action(pre(observation))).squeeze(0).float().cpu().numpy()
            previous = data.ctrl.copy()
            endpoint = bounded_action(raw, previous, model.actuator_ctrlrange)
            contact = samples[-1]
            blocked = bool(
                (contact["fixed_jaw_force_n"] > .01 or contact["moving_jaw_force_n"] > .01)
                and endpoint[5] > previous[5]
                and not stable_support_window(samples[-251:], target_face=2)
            )
            if blocked:
                endpoint[5] = previous[5]
            controls.append({
                "time_s": float(data.time), "raw_action_rad": raw.tolist(),
                "applied_action_rad": endpoint.tolist(), "blocked_open": blocked,
            })
            for step in range(1, 51):
                data.ctrl[:] = interpolate_command(previous, endpoint, step, 50)
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                row, unexpected = capture_sample(
                    model, data, evaluation_phase(cycle * 50 + step),
                    robot_geoms=robot_geoms, distance_cap_m=.05,
                )
                samples.append(row)
                warnings = {
                    mujoco.mjtWarning(i).name: int(w.number)
                    for i, w in enumerate(data.warning) if w.number
                }
                if unexpected:
                    reason = [f"unexpected_contacts:{unexpected}"]
                elif warnings:
                    reason = [f"mujoco_warnings:{warnings}"]
                elif not row["limits_ok"] or row["min_arm_joint_margin_rad"] < .1:
                    reason = ["joint_or_force_limit"]
                if reason:
                    break
            if reason:
                break
            if cycle % 50 == 49:
                print(f"評価 {offset_mm}: {data.time:.1f}s", flush=True)
        if len(samples) >= 9501:
            grasp = validate_grasp(samples[:9501], {}, observe_position=True)
        if len(samples) >= 12501:
            rotation = validate_rotation(samples[:12501], {}, target_face=2, observe_position=True)
        completed = len(samples) == 22501
        if completed:
            placement = validate_placement(samples, {}, target_face=2, observe_position=True)
            reason += [f"{stage}.{key}" for stage, result in
                       (("grasp", grasp), ("rotation", rotation), ("placement", placement))
                       for key in result["failure_reasons"]]
        report = {
            "offset_mm": offset_mm, "seed": seed, "passed": completed and not reason,
            "requested_face": 2, "evaluation_type": evaluation_type,
            "completed": completed, "duration_s": float(data.time),
            "initial_qpos": initial_qpos.tolist(),
            "physics_steps": len(samples) - 1, "failure_reasons": reason,
            "grasp_validation": grasp, "rotation_validation": rotation, "placement_validation": placement,
            "controls": controls, "final_state": samples[-1], "uploaded": False,
            "notice": "評価用位置は学習非使用。時刻窓は評価のみ。画像・関節・指示だけで方策実行。",
        }
        write_json(directory / "evaluation.json", report)
        return report
    finally:
        renderer.close()
        if samples:
            with (directory / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)


def evaluate(directory: Path, training_path: Path, study_path: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    study = json.loads(study_path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(split_positions()))
    if study["split"] != expected:
        raise ValueError("評価位置分割が事前固定した分割と一致しません")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    require_training_contract(training, ENDPOINT_MODE)
    if Path(training["dataset"]).resolve() != Path(study["dataset"]).resolve():
        raise ValueError("評価するモデルが指定した学習分割のデータに対応しません")
    dataset = LeRobotDataset(training["repo_id"], root=Path(training["dataset"]))
    policy, pre, post = load_policy(
        training["models"], dataset, Path(training["checkpoint"]),
        dataset_stats=training_normalization_stats(training, dataset),
    )
    write_json(directory / "settings.json", {
        "study": str(study_path.resolve()), "training": str(training_path.resolve()),
        "study_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "seed": 42, "horizon_s": 45, "split": study["split"],
    })
    results = []
    for index, position in enumerate(study["split"]["evaluation_offsets_mm"]):
        run = directory / f"evaluation-{index:02d}"
        run.mkdir()
        try:
            report = evaluate_position(run, policy, pre, post, position, 42)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            report = {"offset_mm": position, "passed": False, "error": str(error)}
            write_json(run / "error.json", report)
        results.append(report)
        print(f"評価位置 {position}: {report['passed']}", flush=True)
    summary = {"runs": len(results), "passed_count": sum(r["passed"] for r in results),
               "results": results, "uploaded": False}
    write_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="面2VLAを未学習4位置で45秒上限・安全中断あり評価")
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/evaluations"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        report = evaluate(directory, args.training, args.study)
    except (OSError, ValueError, RuntimeError) as error:
        write_json(directory / "error.json", {"message": str(error)})
        print(str(error), file=sys.stderr)
        return 2
    print(f"評価保存先: {directory.resolve()}", flush=True)
    return 0 if report["passed_count"] == report["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
