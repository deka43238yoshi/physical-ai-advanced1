import csv
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(
    all(os.environ.get(key) for key in ("STUDY_REPORT", "STUDY_TRAINING", "STUDY_EVALUATION")),
    "12本学習・4位置評価の生成物を明示した場合だけ実行",
)
class LearningStudyIntegrationTest(unittest.TestCase):
    def test_split_all_training_labels_full_epoch_and_held_out_results(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from so101_dice_sim.action_contract import ENDPOINT_MODE, require_training_contract
        from so101_dice_sim.endpoint_collection import load_endpoint_replay
        from so101_dice_sim.learning import training_batches, bounded_action
        from so101_dice_sim.learning import CAMERAS
        from so101_dice_sim.learning_study import split_positions
        from so101_dice_sim.policy_evaluation import evaluation_phase
        from so101_dice_sim.placement import validate_placement
        from so101_dice_sim.robot_check import JOINT_NAMES
        from so101_dice_sim.robot_check import ROOT
        import mujoco

        study_path = Path(os.environ["STUDY_REPORT"])
        study = json.loads(study_path.read_text(encoding="utf-8"))
        training_path = Path(os.environ["STUDY_TRAINING"])
        training = json.loads(training_path.read_text(encoding="utf-8"))
        evaluation_path = Path(os.environ["STUDY_EVALUATION"])
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        self.assertEqual(study["split"], json.loads(json.dumps(split_positions())))
        self.assertEqual(len(study["results"]), 16)
        self.assertTrue(all(row["passed"] for row in study["results"]))
        contract = require_training_contract(training, ENDPOINT_MODE)
        self.assertEqual(Path(training["dataset"]).resolve(), Path(study["dataset"]).resolve())
        dataset = LeRobotDataset(training["repo_id"], root=Path(training["dataset"]))
        self.assertEqual(len(dataset), 5400)
        self.assertEqual(dataset.meta.total_episodes, 12)
        self.assertEqual(
            {key for key in dataset.features if key.startswith("observation.")},
            {"observation.state", *CAMERAS},
        )
        collection = json.loads(Path(study["collection"]).read_text(encoding="utf-8"))
        values = dataset.hf_dataset.select_columns(["observation.state", "action"])
        train_positions = {tuple(p) for p in study["split"]["train_offsets_mm"]}
        eval_positions = {tuple(p) for p in study["split"]["evaluation_offsets_mm"]}
        self.assertFalse(train_positions & eval_positions)
        for index, episode in enumerate(collection["episodes"]):
            source = Path(episode["source"])
            self.assertTrue(source.name.startswith("train-"))
            for name, digest in episode["source_sha256"].items():
                self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
            replay, actions = load_endpoint_replay(source)
            np.testing.assert_allclose(
                np.asarray(replay.report["initial_qpos"])[6:8],
                np.array([.245, .015]) + np.array(study["split"]["train_offsets_mm"][index]) / 1000,
                atol=1e-12, rtol=0,
            )
            actual = values[index * 450:(index + 1) * 450]
            np.testing.assert_array_equal(actual["action"], actions)
            np.testing.assert_array_equal(
                actual["observation.state"], replay.qpos[0:22500:50, :6].astype(np.float32),
            )
        self.assertEqual(training["epochs"], 1)
        self.assertEqual(training["batch_size"], 4)
        self.assertEqual(training["steps"], 1350)
        self.assertEqual(training["samples_seen"], 5400)
        self.assertTrue(training["monitored_weights_changed"])
        self.assertEqual(
            [row["sample_indices"] for row in training["losses"]], training_batches(5400, 3, 1, 4),
        )
        self.assertTrue(all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
                            for row in training["losses"]))
        settings = json.loads((evaluation_path.parent / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["study_sha256"], hashlib.sha256(study_path.read_bytes()).hexdigest())
        self.assertEqual(settings["training_sha256"], hashlib.sha256(training_path.read_bytes()).hexdigest())
        self.assertEqual(evaluation["runs"], 4)
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
        self.assertEqual(evaluation["passed_count"], sum(row["passed"] for row in evaluation["results"]))
        self.assertEqual({tuple(row["offset_mm"]) for row in evaluation["results"]}, eval_positions)
        for index, result in enumerate(evaluation["results"]):
            self.assertFalse(tuple(result["offset_mm"]) in train_positions)
            self.assertFalse(result["uploaded"])
            self.assertLessEqual(result["duration_s"], 45.00000001)
            path = evaluation_path.parent / f"evaluation-{index:02d}" / "trajectory.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                raw_rows = list(csv.DictReader(stream))
            samples = [{
                key: value if key == "phase" else value == "True" if value in ("True", "False")
                else None if value == "" else int(value) if key == "top_face" else float(value)
                for key, value in row.items()
            } for row in raw_rows]
            self.assertEqual(len(samples), result["physics_steps"] + 1)
            self.assertEqual(samples[-1], result["final_state"])
            np.testing.assert_allclose(
                [samples[0]["dice_x_m"], samples[0]["dice_y_m"]],
                np.array([.245, .015]) + np.array(result["offset_mm"]) / 1000, atol=1e-12,
            )
            for step, row in enumerate(samples):
                self.assertAlmostEqual(row["time_s"], step * .002)
                self.assertEqual(row["phase"], evaluation_phase(step))
                self.assertTrue(np.isfinite([row[f"{name}_target"] for name in JOINT_NAMES]).all())
            if result["passed"]:
                self.assertTrue(result["completed"])
                self.assertTrue(validate_placement(samples, {}, target_face=2, observe_position=True)["passed"])
            else:
                self.assertTrue(result["failure_reasons"])
            self.assertEqual(result["controls"][0]["time_s"], 0)
            targets = np.array([[s[f"{name}_target"] for name in JOINT_NAMES] for s in samples])
            previous = targets[0].copy()
            for cycle, control in enumerate(result["controls"]):
                endpoint = bounded_action(
                    np.array(control["raw_action_rad"]), previous, model.actuator_ctrlrange,
                )
                if control["blocked_open"]:
                    endpoint[5] = previous[5]
                np.testing.assert_array_equal(endpoint, control["applied_action_rad"])
                count = min(50, len(samples) - 1 - cycle * 50)
                expected_targets = previous + np.arange(1, count + 1)[:, None] / 50 * (endpoint - previous)
                if count == 50:
                    expected_targets[-1] = endpoint
                np.testing.assert_allclose(
                    targets[cycle * 50 + 1:cycle * 50 + count + 1],
                    expected_targets, atol=1e-14, rtol=0,
                )
                previous = endpoint
            if any(reason.startswith("unexpected_contacts:") for reason in result["failure_reasons"]):
                self.assertGreater(samples[-1]["unexpected_contacts"], 0)
                self.assertTrue(all(row["unexpected_contacts"] == 0 for row in samples[:-1]))
        self.assertEqual(contract["target_offset_s"], .1)


if __name__ == "__main__":
    unittest.main()
