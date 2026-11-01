import math

import numpy as np
from numpy.typing import NDArray


FACE_AXES = {
    1: (0, 0, 1),
    6: (0, 0, -1),
    2: (1, 0, 0),
    5: (-1, 0, 0),
    3: (0, 1, 0),
    4: (0, -1, 0),
}


def identify_top_face(
    rotation: NDArray[np.float64], tolerance_deg: float
) -> tuple[int | None, float]:
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("面判定には有限の正規直交回転行列が必要です")
    if not math.isfinite(tolerance_deg) or not 0 <= tolerance_deg < 45:
        raise ValueError("面判定の許容角度は0度以上45度未満にしてください")
    scores = {
        number: float(rotation[2] @ np.asarray(axis))
        for number, axis in FACE_AXES.items()
    }
    number = max(scores, key=scores.__getitem__)
    tilt = math.degrees(math.acos(float(np.clip(scores[number], -1, 1))))
    return (number if tilt <= tolerance_deg + 1e-10 else None), tilt
