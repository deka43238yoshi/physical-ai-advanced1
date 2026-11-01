import hashlib
import json
from pathlib import Path


LEGACY_MODE = "next_step_hold_v1"
ENDPOINT_MODE = "endpoint_linear_v2"
ACTION_MODES = (LEGACY_MODE, ENDPOINT_MODE)
CONTRACT_NAME = "action_contract.json"


def action_contract(mode: str) -> dict:
    if mode not in ACTION_MODES:
        raise ValueError("未対応のaction契約です")
    return {
        "version": mode, "fps": 10, "physics_timestep_s": .002,
        "joint_units": "rad", "action_shape": [6],
        "target_offset_s": .1 if mode == ENDPOINT_MODE else .002,
        "execution": "bounded_endpoint_linear_500hz" if mode == ENDPOINT_MODE else "bounded_hold_10hz",
        "interpolation_start": "previous_applied_command",
        "max_endpoint_delta_rad": .05, "arm_margin_rad": .1,
        "airborne_grasp_opening_guard": True,
    }


def dataset_contract(root: Path, expected_mode: str) -> dict:
    path = root / CONTRACT_NAME
    expected = action_contract(expected_mode)
    if not path.exists():
        if expected_mode == LEGACY_MODE:
            return expected
        raise ValueError("次周期末データにはaction_contract.jsonが必要です")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("データセットのaction契約が指定モードと一致しません")
    return actual


def dataset_fingerprint(root: Path) -> dict:
    paths = [root / "meta" / name for name in ("info.json", "stats.json")]
    paths += sorted((root / "data").rglob("*.parquet"))
    if not (root / "data").is_dir() or len(paths) == 2:
        raise ValueError("学習データのParquetがありません")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def require_training_contract(training: dict, mode: str) -> dict:
    root = Path(training["dataset"])
    contract = dataset_contract(root, mode)
    recorded = training.get("action_contract")
    if recorded is None and mode == LEGACY_MODE:
        return contract
    if recorded != contract:
        raise ValueError("学習記録のaction契約が推論モードと一致しません")
    checkpoint = Path(training["checkpoint"])
    saved = json.loads((checkpoint / CONTRACT_NAME).read_text(encoding="utf-8"))
    if saved != contract:
        raise ValueError("チェックポイントのaction契約が一致しません")
    if dataset_fingerprint(root) != training.get("dataset_fingerprint"):
        raise ValueError("学習後にデータまたは正規化統計が変更されています")
    return contract
