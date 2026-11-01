import importlib.util

import numpy as np
import pytest

from so101_dice_sim.image_position import detect_dice_bbox, estimate_xy


def test_missing_or_ambiguous_candidate_is_explicit_error():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="検出数: 0"):
        detect_dice_bbox(image)
    image[10:20, 10:20] = 255
    image[60:70, 60:70] = 255
    with pytest.raises(ValueError, match="検出数: 2"):
        detect_dice_bbox(image)
    image[:20, :20] = 255
    image[60:70, 60:70] = 0
    with pytest.raises(ValueError, match="画像端"):
        detect_dice_bbox(image)


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError, match="校正"):
        estimate_xy(np.zeros((10, 10, 3)), np.zeros(3), np.zeros((3, 3)), 50)


@pytest.mark.skipif(importlib.util.find_spec("PIL") is None, reason="描画イメージで検証")
@pytest.mark.parametrize("xy", [[.240, .010], [.245, .015], [.250, .020]])
def test_known_cube_projection_estimates_without_truth_input(xy):
    from PIL import Image, ImageDraw

    position = np.array([0., -1., 1.4])
    z_axis = position / np.linalg.norm(position)
    rotation = np.column_stack(([1, 0, 0], np.cross(z_axis, [1, 0, 0]), z_axis))
    world = np.array([(x + xy[0], y + xy[1], z) for x in (-.015, .015)
                      for y in (-.015, .015) for z in (0., .03)])
    local = (world - position) @ rotation
    focal = 240 / np.tan(np.deg2rad(25))
    pixels = [tuple(p) for p in ([319.5, 239.5] + focal * local[:, :2] / -local[:, 2, None] * [1, -1])]
    image = Image.new("RGB", (640, 480), (100, 65, 30))
    draw = ImageDraw.Draw(image)
    for face in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)):
        draw.polygon([pixels[i] for i in face], fill=(230, 230, 230))
    result = estimate_xy(np.asarray(image), position, rotation, 50)
    assert np.linalg.norm(np.array(result["estimated_xy_m"]) - xy) < .004
    shifted = np.asarray(image).copy()
    shifted = np.roll(shifted, 5, axis=1)
    other = estimate_xy(shifted, position, rotation, 50)
    assert other["estimated_xy_m"][0] > result["estimated_xy_m"][0] + .01
