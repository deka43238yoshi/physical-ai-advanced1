import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .reporting import create_run_directory, write_json


def detect_dice_bbox(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB画像が必要です")
    values = rgb.astype(float)
    white = (values.min(axis=2) >= 140) & (np.ptp(values, axis=2) <= 35)
    visited = np.zeros(white.shape, dtype=bool)
    components = []
    height, width = white.shape
    for y, x in zip(*np.nonzero(white)):
        if visited[y, x]:
            continue
        pending = [(y, x)]
        visited[y, x] = True
        pixels = []
        while pending:
            row, col = pending.pop()
            pixels.append((row, col))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = row + dy, col + dx
                if 0 <= ny < height and 0 <= nx < width and white[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    pending.append((ny, nx))
        if len(pixels) >= 20:
            points = np.array(pixels)
            components.append(np.array([
                points[:, 1].min(), points[:, 0].min(),
                points[:, 1].max(), points[:, 0].max(),
            ], dtype=float))
    if len(components) != 1:
        raise ValueError(f"白色サイコロ候補は1個必要です。検出数: {len(components)}")
    bbox = components[0]
    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1:
        raise ValueError("候補が画像端で切れています")
    return bbox


def estimate_xy(
    rgb: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray,
    fovy_deg: float,
) -> dict:
    position = np.asarray(camera_position, dtype=float)
    rotation = np.asarray(camera_rotation, dtype=float)
    if (
        position.shape != (3,) or rotation.shape != (3, 3)
        or not np.isfinite(position).all() or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1)
        or not np.isfinite(fovy_deg) or not 0 < fovy_deg < 180
    ):
        raise ValueError("カメラ校正が不正です")
    bbox = detect_dice_bbox(rgb)
    height, width = rgb.shape[:2]
    focal = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    principal = np.array([(width - 1) / 2, (height - 1) / 2])
    corners = np.array([(x, y, z) for x in (-.015, .015)
                        for y in (-.015, .015) for z in (0., .030)])

    def projected_box(xy):
        world = corners + [xy[0], xy[1], 0]
        local = (world - position) @ rotation
        if np.any(local[:, 2] >= 0):
            raise ValueError("サイコロ候補がカメラの背後です")
        pixels = principal + focal * local[:, :2] / -local[:, 2, None] * [1, -1]
        return np.concatenate((pixels.min(axis=0), pixels.max(axis=0)))

    center = (bbox[:2] + bbox[2:]) / 2
    ray = rotation @ np.array([
        (center[0] - principal[0]) / focal, -(center[1] - principal[1]) / focal, -1,
    ])
    if abs(ray[2]) < 1e-9:
        raise ValueError("視線がテーブルと平行です")
    xy = (position + ray * ((.015 - position[2]) / ray[2]))[:2]
    target = bbox + [-.5, -.5, .5, .5]
    for _ in range(30):
        prediction = projected_box(xy)
        residual = prediction - target
        jacobian = np.column_stack([
            (projected_box(xy + delta) - prediction) / 1e-5
            for delta in (np.array([1e-5, 0]), np.array([0, 1e-5]))
        ])
        step = np.linalg.solve(jacobian.T @ jacobian + 1e-8 * np.eye(2), -jacobian.T @ residual)
        xy += step
        if np.linalg.norm(step) < 1e-9:
            break
    residual = float(np.linalg.norm(projected_box(xy) - target))
    if residual > 3 or not np.isfinite(xy).all():
        raise ValueError(f"既知の立方体形状と画像候補が一致しません: {residual:.3f} pixel")
    return {"estimated_xy_m": xy.tolist(), "rgb_bbox_xyxy": bbox.tolist(),
            "bbox_residual_px": residual}


def evaluate(camera_directories: list[Path], directory: Path) -> dict:
    from PIL import Image, ImageDraw

    rows = []
    for source in camera_directories:
        metadata = json.loads((source / "camera_report.json").read_text(encoding="utf-8"))
        if metadata["camera"] != "external_overhead":
            raise ValueError("外部固定カメラ画像が必要です")
        trajectory = Path(metadata["source_directory"]) / "trajectory.csv"
        for name, expected in metadata["source_sha256"].items():
            path = Path(metadata["source_directory"]) / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"出典ハッシュが一致しません: {path}")
        # 正解CSVは評価側だけで読み、検出・推定関数へは渡さない。
        with trajectory.open(newline="", encoding="utf-8") as stream:
            truth = list(csv.DictReader(stream))
        for capture in metadata["captures"]:
            if capture["time_s"] > 2 + 1e-9:
                continue
            row = {"source_directory": str(source), "rgb_file": capture["rgb_file"],
                   "time_s": capture["time_s"], "status": "error"}
            row["rgb_sha256"] = hashlib.sha256((source / capture["rgb_file"]).read_bytes()).hexdigest()
            with Image.open(source / capture["rgb_file"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            try:
                estimate = estimate_xy(
                    rgb, capture["camera_position_world_m"],
                    capture["camera_rotation_world"], metadata["vertical_fov_deg"],
                )
            except ValueError as error:
                row["error"] = str(error)
                print(f"推定失敗: {source.name}/{capture['rgb_file']}: {error}", file=sys.stderr)
            else:
                actual = truth[capture["sample_index"]]
                true_xy = np.array([float(actual[f"dice_{axis}_m"]) for axis in "xy"])
                error = np.array(estimate["estimated_xy_m"]) - true_xy
                row.update(estimate, status="estimated", true_xy_m=true_xy.tolist(),
                           error_xy_mm=(error * 1000).tolist(),
                           error_norm_mm=float(np.linalg.norm(error) * 1000))
                overlay = Image.fromarray(rgb)
                ImageDraw.Draw(overlay).rectangle(estimate["rgb_bbox_xyxy"], outline="red", width=1)
                name = f"{len(rows) + 1:03d}_detection.png"
                overlay.save(directory / name)
                row["detection_file"] = name
            rows.append(row)
    if not rows:
        raise ValueError("0〜2秒の評価画像がありません")
    errors = [row["error_norm_mm"] for row in rows if row["status"] == "estimated"]
    report = {
        "images": len(rows), "estimated_count": len(errors), "failure_count": len(rows) - len(errors),
        "mean_error_mm": float(np.mean(errors)) if errors else None,
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "max_error_mm": max(errors) if errors else None, "results": rows,
        "assumptions": "既知の校正、30mm白色立方体、上面1・方位固定、天板Z=0、着地前後0〜2秒のみ。",
        "notice": "RGBの白色連結成分と既知立方体の投影矩形からXY推定。GTマスク・物体座標は推定入力に使わない。操作には未使用。",
    }
    write_json(directory / "position_estimation.json", report)
    with (directory / "errors.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "source_directory", "rgb_file", "time_s", "status",
            "estimated_x_m", "estimated_y_m", "true_x_m", "true_y_m",
            "error_x_mm", "error_y_mm", "error_norm_mm", "error",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key, "") for key in (
                    "source_directory", "rgb_file", "time_s", "status", "error_norm_mm", "error",
                )},
                **{f"{prefix}_{axis}_{unit}": row[key][i]
                   for prefix, key, unit in (
                       ("estimated", "estimated_xy_m", "m"), ("true", "true_xy_m", "m"),
                       ("error", "error_xy_mm", "mm"),
                   ) if key in row for i, axis in enumerate("xy")},
            })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="固定カメラRGBのXY推定を正解と比較")
    parser.add_argument("--camera-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/image-position"))
    args = parser.parse_args(argv)
    directory = None
    try:
        directory = create_run_directory(args.output_dir)
        report = evaluate(args.camera_dirs, directory)
    except (OSError, ValueError, KeyError) as error:
        print(f"画像位置評価エラー: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {"status": "error", "message": str(error)})
        return 2
    print(f"評価保存先: {directory.resolve()}")
    print(f"推定 {report['estimated_count']}/{report['images']}, RMSE={report['rmse_mm']} mm, 最大={report['max_error_mm']} mm")
    return 0 if not report["failure_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
