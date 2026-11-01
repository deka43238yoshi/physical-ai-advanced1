import argparse
import hashlib
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from .reporting import create_run_directory, write_json
from .visualization import FONT, Replay, ReplayRenderer, add_pips, load_replay, phase_label


DEFAULT_TIMES = (0., 2., 3.5, 5., 6.5, 8.)


def capture_indices(replay: Replay, times: list[float] | tuple[float, ...]) -> list[int]:
    if not times or any(
        not math.isfinite(time) or time < 0 or time > replay.times[-1] + 1e-9
        for time in times
    ):
        raise ValueError("撮影時刻は記録範囲内の有限値を1個以上指定してください")
    return [int(np.argmin(np.abs(replay.times - time))) for time in times]


def capture_wrist(
    replay: Replay, directory: Path, times=DEFAULT_TIMES, *, camera_name: str = "wrist_cam",
) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    indices = capture_indices(replay, times)
    renderer = ReplayRenderer()
    captures = []
    sheet = Image.new("RGB", (1280, 520 * math.ceil(len(indices) / 2)), "#17202b")
    font = ImageFont.truetype(FONT, 18)
    try:
        camera = renderer.model.camera(camera_name).id
        dice_geom = renderer.model.geom("dice_geom").id
        for number, (requested, index) in enumerate(zip(times, indices), 1):
            renderer.data.qpos[:] = replay.qpos[index]
            renderer.data.qvel[:] = 0
            mujoco.mj_forward(renderer.model, renderer.data)
            renderer.renderer.update_scene(
                renderer.data, camera=camera, scene_option=renderer.option,
            )
            add_pips(
                renderer.renderer.scene, renderer.data.xpos[renderer.dice],
                renderer.data.xmat[renderer.dice].reshape(3, 3),
            )
            rgb = renderer.renderer.render().copy()
            # 可視画素は正解形状IDによる診断であり、RGBからの認識結果ではない。
            renderer.renderer.update_scene(
                renderer.data, camera=camera, scene_option=renderer.option,
            )
            renderer.renderer.enable_segmentation_rendering()
            try:
                segmentation = renderer.renderer.render()
            finally:
                renderer.renderer.disable_segmentation_rendering()
            mask = (
                (segmentation[:, :, 0] == dice_geom)
                & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            )
            ys, xs = np.nonzero(mask)
            prefix = f"{number:02d}_{replay.times[index]:06.3f}s"
            Image.fromarray(rgb).save(directory / f"{prefix}_rgb.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{prefix}_mask.png")
            label = phase_label(replay.rows[index]["phase"], replay.report.get("requested_face"))
            x, y = ((number - 1) % 2) * 640, ((number - 1) // 2) * 520
            sheet.paste(Image.fromarray(rgb), (x, y + 40))
            ImageDraw.Draw(sheet).text(
                (x + 8, y + 8),
                f'{replay.times[index]:.3f} 秒  {label}  可視画素: {len(xs)}',
                font=font, fill="white",
            )
            captures.append({
                "requested_time_s": requested, "sample_index": index,
                "time_s": float(replay.times[index]), "phase": replay.rows[index]["phase"],
                "rgb_file": f"{prefix}_rgb.png", "mask_file": f"{prefix}_mask.png",
                "visible_dice_pixels": len(xs),
                "visible_bbox_xyxy": (
                    [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    if len(xs) else None
                ),
                "mask_touches_image_border": bool(
                    len(xs) and (xs.min() == 0 or xs.max() == 639 or ys.min() == 0 or ys.max() == 479)
                ),
                "camera_position_world_m": renderer.data.cam_xpos[camera].tolist(),
                "camera_rotation_world": renderer.data.cam_xmat[camera].reshape(3, 3).tolist(),
            })
        sheet.save(directory / "contact_sheet.jpg", quality=95)
        return {
            "camera": camera_name, "resolution": [640, 480],
            "vertical_fov_deg": float(renderer.model.cam_fovy[camera]),
            "physics_steps": 0, "captures": captures,
            "notice": (
                "保存軌道の名前付きカメラ描画。位置推定・制御入力への利用なし。"
                "マスクは正解形状IDによる可視性診断。目はRGBの描画だけに追加。"
                "照明は既存可視化と同じ補助照明。実カメラの歪み・雑音は未モデル化。"
            ),
        }
    finally:
        renderer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="保存軌道から手首・外部固定カメラの見え方を確認")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera", choices=("wrist_cam", "external_overhead"), default="wrist_cam")
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    args = parser.parse_args(argv)
    directory = None
    try:
        replay = load_replay(args.run_dir)
        capture_indices(replay, args.times)
        output = args.output_dir or Path(
            "outputs/wrist-camera" if args.camera == "wrist_cam" else "outputs/external-camera"
        )
        directory = create_run_directory(output)
        report = capture_wrist(replay, directory, args.times, camera_name=args.camera)
        report.update({
            "source_directory": str(args.run_dir.resolve()),
            "source_validation_passed": replay.report["validation"]["passed"],
            "source_sha256": {
                name: hashlib.sha256((args.run_dir / name).read_bytes()).hexdigest()
                for name in ("result.json", "trajectory.csv")
            },
        })
        write_json(directory / "camera_report.json", report)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"カメラ描画エラー: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {"status": "error", "message": str(error)})
        return 2
    print(f"カメラ保存先: {directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
