import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    scene: str
    duration_s: float
    timestep_s: float
    settle_window_s: float
    linear_speed_limit_m_s: float
    angular_speed_limit_rad_s: float
    height_tolerance_m: float
    face_tolerance_deg: float
    initial_position_m: tuple[float, float, float]
    initial_quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in (
                "scene", "initial_position_m", "initial_quaternion_wxyz"
            ):
                continue
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name}: 有限の正数を指定してください")
        if not isinstance(self.scene, str) or not self.scene:
            raise ValueError("scene: MJCFのパスを指定してください")
        for name, length in (
            ("initial_position_m", 3), ("initial_quaternion_wxyz", 4)
        ):
            vector = getattr(self, name)
            if (
                not isinstance(vector, (list, tuple))
                or len(vector) != length
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in vector
                )
            ):
                raise ValueError(f"{name}: 有限の数値を{length}個指定してください")
        norm = math.sqrt(sum(v * v for v in self.initial_quaternion_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-8):
            raise ValueError("初期クォータニオンは単位長で指定してください")
        if self.settle_window_s > self.duration_s:
            raise ValueError("静止検証時間は実行時間以下にしてください")
        if self.face_tolerance_deg >= 45:
            raise ValueError("上面の許容角度は45度未満にしてください")
        for name in ("duration_s", "settle_window_s"):
            steps = getattr(self, name) / self.timestep_s
            if not math.isfinite(steps) or not math.isclose(
                steps, round(steps), rel_tol=0, abs_tol=1e-8
            ) or steps < 1:
                raise ValueError(f"{name}: 時間刻みの正の整数倍にしてください")

    @property
    def steps(self) -> int:
        return round(self.duration_s / self.timestep_s)

    @property
    def window_steps(self) -> int:
        return round(self.settle_window_s / self.timestep_s)


def load_settings(path: Path) -> Settings:
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("設定はJSONオブジェクトにしてください")
    expected = {field.name for field in fields(Settings)}
    if set(values) != expected:
        raise ValueError(
            f"設定キーが不一致です: 不足={sorted(expected - set(values))}, "
            f"未定義={sorted(set(values) - expected)}"
        )
    settings = Settings(**values)
    resolved = (path.resolve().parent / settings.scene).resolve()
    return Settings(**{
        **values,
        "scene": str(resolved),
        "initial_position_m": tuple(settings.initial_position_m),
        "initial_quaternion_wxyz": tuple(settings.initial_quaternion_wxyz),
    })
