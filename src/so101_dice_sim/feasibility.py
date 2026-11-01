import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from .faces import FACE_AXES, identify_top_face
from .grasp import contact_state
from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES, ROOT, SOURCE_COMMIT, verify_assets


def survey_faces(source: Path, directory: Path) -> dict:
    provenance = verify_assets(ROOT / "assets" / "robotstudio_so101")
    with (source / "result.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    if (
        report.get("task") != "so101_fixed_place_face_2"
        or report.get("source_commit") != SOURCE_COMMIT
        or report.get("mujoco_version") != mujoco.__version__
        or not report.get("validation", {}).get("passed")
    ):
        raise ValueError("調査元には同じ資産・MuJoCoで成功した固定上面2配置結果が必要です")
    with (source / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != report["sample_count"] or len(rows) <= 8000:
        raise ValueError("調査元CSVのサンプル数が一致しません")
    row = rows[8000]
    if not math.isclose(float(row["time_s"]), 16, abs_tol=1e-9):
        raise ValueError("調査元CSVの16秒保持状態を取得できません")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:6] = [float(row[f"{name}_qpos"]) for name in JOINT_NAMES]
    data.qpos[6:9] = [float(row[f"dice_{axis}_m"]) for axis in "xyz"]
    data.qpos[9:13] = [float(row[f"dice_quat_{axis}"]) for axis in "wxyz"]
    if not np.isfinite(data.qpos).all():
        raise ValueError("保持状態に非有限値があります")
    mujoco.mj_forward(model, data)
    gripper, dice = model.body("gripper").id, model.body("dice").id
    rotation = data.xmat[gripper].reshape(3, 3).copy()
    relative_position = rotation.T @ (data.xpos[dice] - data.xpos[gripper])
    relative_rotation = rotation.T @ data.xmat[dice].reshape(3, 3)
    initial = data.qpos[:6].copy()
    planning = mujoco.MjData(model)
    low, high = model.jnt_range[:5, 0] + .1, model.jnt_range[:5, 1] - .1
    xs = (.245, .30, .36, .392)
    heights = (.06, .10, .23)

    def pose(q):
        planning.qpos[:6] = q
        mujoco.mj_forward(model, planning)
        r = planning.xmat[gripper].reshape(3, 3)
        return planning.xpos[gripper] + r @ relative_position, r @ relative_rotation

    def solve(face, x, height, seed):
        q = seed.copy()
        normal = np.asarray(FACE_AXES[face])

        def error(angles):
            position, r = pose(angles)
            return np.concatenate((
                np.array([x, height]) - position[[0, 2]],
                .05 * (np.array([0, 0, 1]) - r @ normal),
            ))

        for _ in range(160):
            residual = error(q)
            if (
                np.linalg.norm(residual[:2]) < .0001
                and np.linalg.norm(residual[2:]) < .05 * math.sin(math.radians(1))
            ):
                return q
            jacobian = np.empty((5, 5))
            for col in range(5):
                perturbed = q.copy()
                perturbed[col] += 1e-6
                jacobian[:, col] = (error(perturbed) - residual) / 1e-6
            delta = -jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-6 * np.eye(5), residual
            )
            q[:5] = np.clip(q[:5] + np.clip(delta, -.15, .15), low, high)
        return None

    def geometry(q):
        center, r = pose(q)
        planning.qpos[6:9] = center
        mujoco.mju_mat2Quat(planning.qpos[9:13], r.ravel())
        mujoco.mj_forward(model, planning)
        _, unexpected = contact_state(model, planning)
        face, tilt = identify_top_face(r, 5)
        bottom = float(center[2] - (np.abs(r) @ np.full(3, .015))[2])
        return {
            "center_m": center.tolist(), "top_face": face, "tilt_deg": tilt,
            "bottom_gap_m": bottom,
            "unexpected_contact_pairs": [
                [model.geom(g).name or f"geom-{g}" for g in pair] for pair in unexpected
            ],
        }

    faces = []
    for face in range(1, 7):
        attempts = []
        for x in xs:
            for height in heights:
                for flex, roll in (
                    (-1.4, -1.52), (0, -1.52), (1.4, -1.52),
                    (0, -2.4), (0, 0), (0, 1.2), (0, 2.4),
                ):
                    seed = initial.copy()
                    seed[3:5] = [flex, roll]
                    hold = solve(face, x, height, seed)
                    if hold is None:
                        continue
                    touch = solve(face, x, .015, hold)
                    hold_geometry = geometry(hold)
                    rotation_path = [
                        geometry(initial + t * (hold - initial))
                        for t in np.linspace(0, 1, 51)
                    ]
                    lower_path = [] if touch is None else [
                        geometry(hold + t * (touch - hold)) for t in np.linspace(0, 1, 51)
                    ]
                    blockers = []
                    if any(s["unexpected_contact_pairs"] for s in rotation_path):
                        blockers.append("回転経路で想定外接触")
                    if any(s["bottom_gap_m"] < .03 for s in rotation_path):
                        blockers.append("回転経路で下面高さ30 mm未満")
                    if touch is None:
                        blockers.append("配置姿勢IK未収束")
                    elif any(s["unexpected_contact_pairs"] for s in lower_path):
                        blockers.append("下降経路で想定外接触")
                    elif any(s["bottom_gap_m"] < -.001 for s in lower_path):
                        blockers.append("下降経路で天板への過大貫入")
                    attempts.append({
                        "x_m": x, "hold_center_height_m": height,
                        "hold_joint_target_rad": hold.tolist(),
                        "touch_joint_target_rad": None if touch is None else touch.tolist(),
                        "hold_geometry": hold_geometry,
                        "touch_geometry": None if touch is None else geometry(touch),
                        "blockers": blockers,
                        "rotation_path_min_bottom_gap_m": min(s["bottom_gap_m"] for s in rotation_path),
                        "lower_path_unexpected_contacts": [
                            s["unexpected_contact_pairs"] for s in lower_path
                            if s["unexpected_contact_pairs"]
                        ][:3],
                    })
        clear = [a for a in attempts if not a["blockers"]]
        faces.append({
            "target_face": face, "orientation_ik_candidates": len(attempts),
            "kinematic_clear_candidates": len(clear),
            "status": "kinematic_candidate" if clear else "no_clear_candidate_in_search",
            "blocker_counts": {
                reason: sum(reason in candidate["blockers"] for candidate in attempts)
                for reason in sorted({reason for a in attempts for reason in a["blockers"]})
            },
            "physical_placement_verified": face == 2,
            "evidence": "面2は調査元の物理実行で確認済み。他面の剛体把持仮定は物理成功を意味しない。",
            "candidates": attempts,
        })
        print(f"面{face}: 姿勢候補{len(attempts)}、経路干渉なし候補{len(clear)}", flush=True)
    result = {
        "source_directory": str(source.resolve()), "source_commit": provenance["commit"],
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("result.json", "trajectory.csv")
        },
        "mujoco_version": mujoco.__version__,
        "method": "16秒の実測把持を剛体保持と仮定した独立計画データの減衰最小二乗IKと経路接触調査",
        "search": {
            "initial_position_m": report["initial_qpos"][6:9], "initial_top_face": 1,
            "x_candidates_m": list(xs), "hold_center_heights_m": list(heights),
            "touch_center_height_m": .015, "seeds_per_x_and_height": 7,
            "position_tolerance_m": .0001, "face_axis_tolerance_deg": 1,
            "max_iterations": 160, "path_samples_per_segment": 51,
            "required_joint_margin_rad": .1,
        },
        "limitations": [
            "限定的な多初期値探索。候補なしは物理的に不可能という証明ではない。",
            "接触・滑りを含む動力学と保持力、解放・退避はこの調査では検証しない。",
            "姿勢・X/Zだけを拘束しYは機構で決まる。補間経路での面維持は要求しない。",
            "再把持・把持方向変更・障害物回避の探索は含まない。基部の衝突形状は上流にない。",
            "候補数は収束した試行数で、異なる初期値が同じ解に収束した重複を含む。51点の離散干渉検査で連続安全を保証しない。",
        ],
        "faces": faces,
    }
    write_json(directory / "feasibility.json", result)
    with (directory / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["目標面", "姿勢IK候補数", "経路干渉なし候補数", "探索状態", "物理配置確認"])
        for face in faces:
            writer.writerow([
                face["target_face"], face["orientation_ik_candidates"],
                face["kinematic_clear_candidates"], face["status"],
                face["physical_placement_verified"],
            ])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定横把持から全6面の限定的な到達・配置調査")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/face-surveys"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        survey_faces(args.run_dir, directory)
    except (OSError, ValueError, KeyError) as error:
        print(f"面調査エラー: {error}", file=sys.stderr)
        write_json(directory / "error.json", {"status": "error", "message": str(error)})
        return 2
    print(f"面調査保存先: {directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
