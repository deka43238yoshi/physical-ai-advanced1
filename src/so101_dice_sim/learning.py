import argparse
import csv
import hashlib
import json
from pathlib import Path
import ssl
import sys
import time

import mujoco
import numpy as np

from .reporting import create_run_directory, write_json
from .robot_check import JOINT_NAMES
from .action_contract import (
    ACTION_MODES, CONTRACT_NAME, ENDPOINT_MODE, LEGACY_MODE,
    dataset_contract, dataset_fingerprint, require_training_contract,
)


TASK = "Pick up the dice and place it with face 2 facing up."
CAMERAS = {"observation.images.camera1": "external_overhead",
           "observation.images.camera2": "wrist_cam"}
FPS = 10


def training_batches(sample_count: int, steps: int, epochs: int | None, batch_size: int) -> list[list[int]]:
    if sample_count < 1:
        raise ValueError("学習データが空です")
    if epochs is None:
        return [[(step * sample_count // steps + 25) % sample_count] for step in range(steps)]
    generator = np.random.default_rng(42)
    batches = []
    for _ in range(epochs):
        order = generator.permutation(sample_count).tolist()
        batches.extend(order[start:start + batch_size] for start in range(0, len(order), batch_size))
    return batches


def build_ca_bundle(default: Path, extra: Path, output: Path) -> str:
    certificate = extra.read_bytes()
    if b"-----BEGIN CERTIFICATE-----" in certificate:
        pem = certificate.decode("ascii")
        der = ssl.PEM_cert_to_DER_cert(pem)
    else:
        der = certificate
        pem = ssl.DER_cert_to_PEM_cert(der)
    output.write_text(default.read_text(encoding="ascii") + "\n" + pem, encoding="ascii")
    ssl.create_default_context(cafile=str(output))
    return hashlib.sha256(der).hexdigest()


def teacher_indices(sample_count: int, timestep_s: float) -> range:
    if sample_count < 2 or not np.isfinite(timestep_s) or timestep_s <= 0:
        raise ValueError("教師軌道のサンプル数・時間刻みが不正です")
    step = round(1 / FPS / timestep_s)
    if step < 1 or not np.isclose(step * timestep_s, 1 / FPS):
        raise ValueError("記録刻みと収集周期が整合しません")
    return range(0, sample_count - 1, step)


def bounded_action(action: np.ndarray, previous: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    if (
        action.shape != (6,) or previous.shape != (6,) or ranges.shape != (6, 2)
        or not all(np.isfinite(value).all() for value in (action, previous, ranges))
    ):
        raise ValueError("推論指令の形状または値が不正です")
    lower, upper = ranges[:, 0].astype(float), ranges[:, 1].astype(float)
    lower[:5] += .1
    upper[:5] -= .1
    if np.any(lower > upper) or np.any(previous < lower) or np.any(previous > upper):
        raise ValueError("前回指令または関節範囲が安全制限の外です")
    bounded = np.clip(action, lower, upper)
    return previous + np.clip(bounded - previous, -.05, .05)


def gpu_check(directory: Path) -> dict:
    import torch
    import importlib.metadata

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUを利用できません。CPU学習へは切り替えません")
    tensor = torch.randn((256, 256), device="cuda")
    result = tensor @ tensor.T
    torch.cuda.synchronize()
    if not torch.isfinite(result).all():
        raise RuntimeError("GPU行列計算が非有限値になりました")
    report = {
        "gpu": torch.cuda.get_device_name(), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "lerobot": importlib.metadata.version("lerobot"), "matrix_test_passed": True,
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
    }
    write_json(directory / "gpu.json", report)
    return report


def download_models(directory: Path, ca_cert: Path | None = None) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    if ca_cert is not None:
        import certifi
        import requests
        from huggingface_hub import configure_http_backend

        bundle = directory / "download-ca-bundle.pem"
        fingerprint = build_ca_bundle(Path(certifi.where()), ca_cert, bundle)
        write_json(directory / "download-trust.json", {
            "additional_ca_sha256": fingerprint, "tls_verification": True,
            "scope": "このダウンロードプロセスだけ。OSの信頼ストアは変更しない。",
        })

        def backend():
            session = requests.Session()
            session.verify = str(bundle.resolve())
            return session

        configure_http_backend(backend_factory=backend)
    models = {}
    for label, repo in (("policy", "lerobot/smolvla_base"),
                        ("backbone", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")):
        info = HfApi(token=False).model_info(repo)
        revision = info.sha
        path = snapshot_download(
            repo, revision=revision, token=False,
            allow_patterns=(["*.json", "*.safetensors"] if label == "policy"
                            else ["*.json", "*.txt", "*.model", "*.jinja"]),
        )
        models[label] = {
            "repo": repo, "revision": revision, "path": path,
            "license": info.card_data.get("license") if info.card_data is not None else None,
        }
    write_json(directory / "models.json", models)
    return models


def render_observation(renderer, qpos: np.ndarray) -> dict[str, np.ndarray]:
    from PIL import Image, ImageOps
    from .visualization import add_pips

    renderer.data.qpos[:] = qpos
    mujoco.mj_forward(renderer.model, renderer.data)
    result = {}
    for key, name in CAMERAS.items():
        renderer.renderer.update_scene(renderer.data, camera=name, scene_option=renderer.option)
        add_pips(renderer.renderer.scene, renderer.data.xpos[renderer.dice],
                 renderer.data.xmat[renderer.dice].reshape(3, 3))
        image = Image.fromarray(renderer.renderer.render())
        result[key] = np.asarray(ImageOps.pad(image, (256, 256), color="black")).copy()
    return result


def collect(directory: Path, sources: list[Path]) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .visualization import ReplayRenderer, load_replay

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        **{key: {"dtype": "image", "shape": (256, 256, 3),
                 "names": ["height", "width", "channels"]} for key in CAMERAS},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/so101_face2_smoke", root=directory / "dataset",
        fps=FPS, robot_type="so101_mujoco_rad", features=features, use_videos=False,
    )
    renderer = ReplayRenderer()
    episodes = []
    try:
        for source in sources:
            replay = load_replay(source)
            if replay.report.get("requested_face") != 2 or not replay.report["validation"]["passed"]:
                raise ValueError(f"成功した面2の記録が必要です: {source}")
            indices = teacher_indices(len(replay.rows), replay.report["timestep_s"])
            for index in indices:
                row = replay.rows[index + 1]
                dataset.add_frame({
                    "observation.state": replay.qpos[index, :6].astype(np.float32),
                    "action": np.array([float(row[f"{name}_target"]) for name in JOINT_NAMES], dtype=np.float32),
                    **render_observation(renderer, replay.qpos[index]), "task": TASK,
                })
            dataset.save_episode()
            episodes.append({
                "source": str(source.resolve()), "frames": len(indices),
                "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ("result.json", "trajectory.csv")},
            })
            print(f"教師エピソード保存: {source.name}", flush=True)
        dataset.finalize()
    finally:
        renderer.close()
    report = {
        "dataset": str((directory / "dataset").resolve()), "episodes": episodes,
        "fps": FPS, "joint_units": "rad", "action": "次の物理ステップでの絶対関節指令",
        "cameras": CAMERAS, "task": TASK, "uploaded": False,
        "notice": "正解座標教師の成功記録を再描画。10Hzへ間引き、教師500Hzの完全再現ではない。少数配管試験。",
    }
    write_json(directory / "collection.json", report)
    return report


def training_normalization_stats(training: dict, dataset) -> dict:
    recorded = training.get("normalization_stats")
    if recorded is None:
        return dataset.meta.stats
    import numpy as np

    stats = {}
    for key in ("observation.state", "action"):
        if key not in recorded:
            raise ValueError(f"学習記録に正規化統計がありません: {key}")
        values = recorded[key]
        mean = np.asarray(values.get("mean"), dtype=np.float32)
        std = np.asarray(values.get("std"), dtype=np.float32)
        if (
            mean.shape != (6,) or std.shape != (6,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError(f"学習記録の正規化統計が不正です: {key}")
        stats[key] = {"mean": mean, "std": std}
    return stats


def load_policy(
    models: dict, dataset, checkpoint: Path | None = None, *, dataset_stats: dict | None = None,
):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    path = str(checkpoint or models["policy"]["path"])
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("SmolVLAの事前学習済み設定が必要です")
    config.device = "cuda"
    config.push_to_hub = False
    config.load_vlm_weights = False
    config.vlm_model_name = models["backbone"]["path"]
    config.chunk_size = 10
    config.n_action_steps = 1
    config.resize_imgs_with_padding = (256, 256)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for key in CAMERAS},
    }
    policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True, local_files_only=True).to("cuda")
    pre, post = make_smolvla_pre_post_processors(
        config, dataset.meta.stats if dataset_stats is None else dataset_stats,
    )
    return policy, pre, post


def checkpoint_fingerprint(checkpoint: Path) -> dict[str, str]:
    if not checkpoint.is_dir():
        raise ValueError("継続学習元のチェックポイントがありません")
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not files:
        raise ValueError("継続学習元のチェックポイントが空です")
    return {
        path.relative_to(checkpoint).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def resolve_resume_training(
    training_path: Path, dataset_path: Path, models_path: Path | None,
    action_mode: str, epochs: int | None,
) -> tuple[dict, Path, dict[str, str]]:
    if action_mode != ENDPOINT_MODE or epochs != 1:
        raise ValueError("継続学習はendpoint契約で1epochだけ許可されます")
    previous = json.loads(training_path.read_text(encoding="utf-8"))
    require_training_contract(previous, action_mode)
    if (
        previous.get("epochs") != 1
        or previous.get("initialized_from_training") is not None
        or Path(previous["dataset"]).resolve() != dataset_path.resolve()
        or type(previous.get("samples_seen")) is not int
        or previous["samples_seen"] < 1
        or type(previous.get("steps")) is not int
        or previous["steps"] < 1
    ):
        raise ValueError("継続学習は同じデータセットの完了済み1epoch記録からのみ可能です")
    if models_path is not None and json.loads(models_path.read_text(encoding="utf-8")) != previous["models"]:
        raise ValueError("継続学習のbase model定義が一致しません")
    checkpoint = Path(previous["checkpoint"])
    fingerprint = checkpoint_fingerprint(checkpoint)
    if not previous.get("steps") or not previous.get("monitored_weights_changed"):
        raise ValueError("有効な重み更新を含む学習元チェックポイントが必要です")
    return previous, checkpoint, fingerprint


def train(
    directory: Path, dataset_path: Path, models_path: Path | None, steps: int,
    action_mode: str = LEGACY_MODE, *, epochs: int | None = None, batch_size: int = 1,
    resume_training_path: Path | None = None,
) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if epochs is None and not 1 <= steps <= 100:
        raise ValueError("最小学習のstepsは1〜100で指定してください")
    if epochs is not None and (type(epochs) is not int or not 1 <= epochs <= 10):
        raise ValueError("本学習のepochsは1〜10で指定してください")
    if type(batch_size) is not int or not 1 <= batch_size <= 4:
        raise ValueError("batch-sizeは1〜4で指定してください")
    contract = dataset_contract(dataset_path, action_mode)
    fingerprint = dataset_fingerprint(dataset_path)
    gpu_check(directory)
    torch.manual_seed(42)
    previous_training = None
    source_checkpoint = None
    source_checkpoint_fingerprint = None
    if resume_training_path is not None:
        previous_training, source_checkpoint, source_checkpoint_fingerprint = resolve_resume_training(
            resume_training_path, dataset_path, models_path, action_mode, epochs,
        )
        if previous_training["dataset_fingerprint"] != fingerprint:
            raise ValueError("継続学習元と現在のデータ/統計指紋が一致しません")
        models = previous_training["models"]
    else:
        if models_path is None:
            raise ValueError("新規学習にはbase models.jsonが必要です")
        models = json.loads(models_path.read_text(encoding="utf-8"))
    repo_id = "local/so101_face2_endpoint_v2" if action_mode == ENDPOINT_MODE else "local/so101_face2_smoke"
    dataset = LeRobotDataset(
        repo_id, root=dataset_path,
        delta_timestamps={"action": [i / FPS for i in range(10)]},
    )
    if previous_training is not None and previous_training["samples_seen"] != len(dataset):
        raise ValueError("継続学習元のepochがデータセット全体を一度ずつ使用していません")
    policy, pre, _ = load_policy(models, dataset, source_checkpoint)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=policy.config.optimizer_lr, betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps, weight_decay=policy.config.optimizer_weight_decay,
    )
    monitored = next(p for p in policy.parameters() if p.requires_grad)
    before = monitored.detach().clone()
    losses = []
    start = time.perf_counter()
    policy.train()
    batches = training_batches(len(dataset), steps, epochs, batch_size)
    steps = len(batches)
    for step, indices in enumerate(batches):
        index = indices[0]
        batch = pre(torch.utils.data.default_collate([dataset[i] for i in indices]))
        if "action_is_pad" in batch:
            batch["actions_id_pad"] = batch["action_is_pad"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("学習lossが非有限値になりました")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        losses.append({"step": step + 1, "sample_index": index,
                       **({"sample_indices": indices} if epochs is not None else {}),
                       "loss": float(loss.detach()), "grad_norm": float(norm)})
        if epochs is None or step % 25 == 0 or step + 1 == steps:
            print(f"学習 {step + 1}/{steps}: loss={losses[-1]['loss']:.6f}", flush=True)
    checkpoint = directory / "checkpoint"
    changed = bool(torch.any(monitored.detach() != before))
    if not changed:
        raise RuntimeError("最小学習で監視対象の重みが更新されていません")
    policy.save_pretrained(checkpoint)
    write_json(checkpoint / CONTRACT_NAME, contract)
    report = {
        "steps": steps, "losses": losses, "elapsed_s": time.perf_counter() - start,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "dataset": str(dataset_path.resolve()), "models": models,
        "checkpoint": str(checkpoint.resolve()), "uploaded": False,
        "monitored_weights_changed": changed,
        "pretrained_weights_strict": True, "seed": 42,
        "action_contract": contract, "dataset_fingerprint": fingerprint, "repo_id": repo_id,
        "epochs": epochs, "batch_size": batch_size if epochs is not None else 1,
        "samples_seen": sum(len(indices) for indices in batches),
        "initialized_from_training": str(resume_training_path.resolve()) if resume_training_path else None,
        "source_checkpoint_fingerprint": source_checkpoint_fingerprint,
        "optimizer_state_restored": False if previous_training is not None else None,
        "notice": (
            "既存checkpointの重みから1epoch継続。optimizer状態は保存されていないため再初期化。"
            if previous_training is not None else "性能は別の物理評価が必要。optimizer再開用保存なし。"
        ),
    }
    write_json(directory / "training.json", report)
    return report


def rollout(
    directory: Path, training_path: Path, seconds: float, action_mode: str = LEGACY_MODE,
) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .grasp import contact_state
    from .placement import PLACE_INITIAL_POSITION_M
    from .visualization import ReplayRenderer
    from .teacher_replay import interpolate_command

    if not np.isfinite(seconds) or not 0 < seconds <= 5:
        raise ValueError("最小推論の時間は0より大きく5秒以下です")
    if action_mode == ENDPOINT_MODE and not np.isclose(seconds * FPS, round(seconds * FPS)):
        raise ValueError("次周期末推論の時間は0.1秒単位で指定してください")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    contract = require_training_contract(training, action_mode)
    dataset = LeRobotDataset(
        training.get("repo_id", "local/so101_face2_smoke"), root=Path(training["dataset"]),
    )
    policy, pre, post = load_policy(
        training["models"], dataset, Path(training["checkpoint"]),
        dataset_stats=training_normalization_stats(training, dataset),
    )
    policy.eval()
    policy.reset()
    renderer = ReplayRenderer()
    model = renderer.model
    if not np.isclose(model.opt.timestep, contract["physics_timestep_s"], atol=1e-12, rtol=0):
        renderer.close()
        raise ValueError("物理時間刻みがaction契約と一致しません")
    data = mujoco.MjData(model)
    data.qpos[5] = .6
    data.qpos[6:9] = PLACE_INITIAL_POSITION_M
    data.ctrl[:] = data.qpos[:6]
    model.actuator_forcerange[5] = [-.25, .25]
    mujoco.mj_forward(model, data)
    rows = []
    physics_rows = []
    torch.manual_seed(42)
    steps_per_control = round(1 / FPS / model.opt.timestep)
    if not np.isclose(steps_per_control * model.opt.timestep, 1 / FPS):
        renderer.close()
        raise ValueError("制御周期と物理時間刻みが一致しません")
    try:
        for i in range(round(seconds * FPS)):
            observation = {
                key: torch.from_numpy(image).permute(2, 0, 1).float() / 255
                for key, image in render_observation(renderer, data.qpos).items()
            }
            observation.update({"observation.state": torch.tensor(data.qpos[:6], dtype=torch.float32),
                                "task": TASK})
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action = post(policy.select_action(pre(observation))).squeeze(0).float().cpu().numpy()
            inference_wall_s = time.perf_counter() - started
            safe = bounded_action(action, data.ctrl, model.actuator_ctrlrange)
            contact, _ = contact_state(model, data)
            blocked_open = bool(
                (contact["fixed_jaw_force_n"] > .01 or contact["moving_jaw_force_n"] > .01)
                and contact["dice_table_force_n"] <= .01 and safe[5] > data.ctrl[5]
            )
            if blocked_open:
                safe[5] = data.ctrl[5]
            previous = data.ctrl.copy()
            for step in range(1, steps_per_control + 1):
                data.ctrl[:] = (
                    interpolate_command(previous, safe, step, steps_per_control)
                    if action_mode == ENDPOINT_MODE else safe
                )
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("推論物理ステップで非有限状態が発生")
                contacts, unexpected = contact_state(model, data)
                if contacts["unexpected_contacts"]:
                    pairs = [
                        [model.geom(g).name or f"geom-{g}" for g in pair] for pair in unexpected
                    ]
                    raise RuntimeError(f"推論物理ステップで想定外接触を検出: {pairs}")
                if action_mode == ENDPOINT_MODE:
                    physics_rows.append({
                        "time_s": float(data.time), "control_index": i,
                        **{f"{name}_target": float(data.ctrl[j]) for j, name in enumerate(JOINT_NAMES)},
                        **{f"{name}_qpos": float(data.qpos[j]) for j, name in enumerate(JOINT_NAMES)},
                        "dice_x_m": float(data.qpos[6]), "dice_y_m": float(data.qpos[7]),
                        "dice_z_m": float(data.qpos[8]),
                    })
            rows.append({
                "time_s": float(data.time), "raw_action_rad": action.tolist(),
                "applied_action_rad": safe.tolist(), "joint_qpos_rad": data.qpos[:6].tolist(),
                "safety_adjusted": bool(not np.allclose(action, safe)),
                "blocked_open": blocked_open, "inference_wall_s": inference_wall_s,
                "control_and_physics_wall_s": time.perf_counter() - started,
                **({"start_action_rad": previous.tolist(), "target_time_s": (i + 1) / FPS}
                   if action_mode == ENDPOINT_MODE else {}),
            })
            print(f"MuJoCo推論 {i + 1}/{round(seconds * FPS)}", flush=True)
        report = {
            "controls": len(rows), "simulated_s": float(data.time),
            "physics_steps": round(seconds * FPS) * round(1 / FPS / model.opt.timestep),
            "rows": rows, "training": str(training_path.resolve()),
            "task_success_evaluated": False, "uploaded": False,
            "action_contract": contract, "seed": 42,
            "safety": "絶対関節範囲、腕余裕0.1rad、指令変化0.05rad/周期、空中把持時の開指抑止。補正全記録。",
            "notice": "短時間の推論接続試験。実時間速度や面2配置成功は未検証。物体状態の上書きは初期化だけ。",
        }
        write_json(directory / "rollout.json", report)
    except (RuntimeError, ValueError) as error:
        write_json(directory / "rollout_abort.json", {
            "completed": False, "reason": str(error), "simulated_s": float(data.time),
            "completed_controls": len(rows), "rows": rows,
            "action_contract": contract, "training": str(training_path.resolve()),
            "last_qpos": data.qpos.tolist(), "last_applied_command_rad": data.ctrl.tolist(),
            "task_success_evaluated": False, "uploaded": False,
        })
        raise
    finally:
        renderer.close()
        if physics_rows:
            with (directory / "physics.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(physics_rows[0]))
                writer.writeheader()
                writer.writerows(physics_rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ローカルSO-101 SmolVLA最小実証")
    parser.add_argument("stage", choices=("gpu", "download", "collect", "collect-endpoints", "train", "rollout"))
    parser.add_argument("--sources", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--resume-training", type=Path, help="同一データで完了済み1epochの学習記録から1epochだけ継続")
    parser.add_argument("--ca-cert", type=Path, help="download時だけ信頼済み公開CA（DER/PEM）を追加")
    parser.add_argument("--training", type=Path)
    training_length = parser.add_mutually_exclusive_group()
    training_length.add_argument("--steps", type=int, default=3)
    training_length.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-mode", choices=ACTION_MODES, default=LEGACY_MODE)
    parser.add_argument("--seconds", type=float, default=2.)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/runs"))
    args = parser.parse_args(argv)
    if args.epochs is not None and args.stage != "train":
        parser.error("--epochsはtrainでだけ使用できます")
    if args.resume_training is not None and args.stage != "train":
        parser.error("--resume-trainingはtrainでだけ使用できます")
    if args.batch_size != 1 and (args.stage != "train" or args.epochs is None):
        parser.error("--batch-sizeはtrain --epochsと組み合わせて指定してください")
    if args.ca_cert is not None and args.stage != "download":
        parser.error("--ca-certはdownloadでだけ使用できます")
    if args.stage not in ("train", "rollout") and args.action_mode != LEGACY_MODE:
        parser.error("--action-modeはtrain/rolloutでだけ指定できます")
    required = {
        "collect": ("sources",), "collect-endpoints": ("sources",),
        "train": ("dataset",), "rollout": ("training",),
    }
    for key in required.get(args.stage, ()):
        if not getattr(args, key):
            parser.error(f"{args.stage}には--{key}が必要です")
    if args.stage == "train":
        if not args.dataset:
            parser.error("trainには--datasetが必要です")
        if not args.models and not args.resume_training:
            parser.error("trainには--modelsまたは--resume-trainingが必要です")
    if args.resume_training is not None and (
        args.epochs != 1 or args.batch_size != 4 or args.action_mode != ENDPOINT_MODE
    ):
        parser.error("--resume-trainingはendpoint_linear_v2・--epochs 1・--batch-size 4に限定されます")
    directory = None
    try:
        directory = create_run_directory(args.output_dir)
        if args.stage == "gpu":
            gpu_check(directory)
        elif args.stage == "download":
            download_models(directory, args.ca_cert)
        elif args.stage == "collect":
            collect(directory, args.sources)
        elif args.stage == "collect-endpoints":
            from .endpoint_collection import collect_endpoints
            collect_endpoints(directory, args.sources)
        elif args.stage == "train":
            if args.epochs is None and args.batch_size == 1:
                train(directory, args.dataset, args.models, args.steps, args.action_mode)
            elif args.resume_training is not None:
                train(
                    directory, args.dataset, args.models, args.steps, args.action_mode,
                    epochs=args.epochs, batch_size=args.batch_size,
                    resume_training_path=args.resume_training,
                )
            else:
                train(directory, args.dataset, args.models, args.steps, args.action_mode,
                      epochs=args.epochs, batch_size=args.batch_size)
        else:
            rollout(directory, args.training, args.seconds, args.action_mode)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"学習実証エラー: {error}", file=sys.stderr)
        if directory is not None:
            write_json(directory / "error.json", {"status": "error", "type": type(error).__name__, "message": str(error)})
        return 2
    print(f"学習実証保存先: {directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
