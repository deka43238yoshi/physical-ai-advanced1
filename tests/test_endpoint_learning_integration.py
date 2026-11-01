import csv
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(
    all(os.environ.get(key) for key in ("ENDPOINT_COLLECTION", "ENDPOINT_TRAINING", "ENDPOINT_ROLLOUT")),
    "次周期末データ・学習・推論の3つのJSONを明示した場合だけ実行",
)
class EndpointLearningIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        cls.collection = json.loads(Path(os.environ["ENDPOINT_COLLECTION"]).read_text(encoding="utf-8"))
        cls.training = json.loads(Path(os.environ["ENDPOINT_TRAINING"]).read_text(encoding="utf-8"))
        cls.rollout_path = Path(os.environ["ENDPOINT_ROLLOUT"])
        cls.rollout = json.loads(cls.rollout_path.read_text(encoding="utf-8"))
        cls.dataset = LeRobotDataset(
            cls.collection["repo_id"], root=Path(cls.collection["dataset"]),
            delta_timestamps={"action": [i / 10 for i in range(10)]},
        )

    def test_all_labels_actual_states_and_rendered_images(self):
        from so101_dice_sim.action_contract import ENDPOINT_MODE, action_contract
        from so101_dice_sim.endpoint_collection import load_endpoint_replay
        from so101_dice_sim.learning import CAMERAS, render_observation
        from so101_dice_sim.visualization import ReplayRenderer

        self.assertEqual(self.collection["action_contract"], action_contract(ENDPOINT_MODE))
        self.assertEqual(self.dataset.meta.total_episodes, 3)
        self.assertEqual(len(self.dataset), 1350)
        self.assertEqual(
            {key for key in self.dataset.features if key.startswith("observation.")},
            {"observation.state", *CAMERAS},
        )
        values = self.dataset.hf_dataset.select_columns(["observation.state", "action"])
        renderer = ReplayRenderer()
        try:
            for episode_index, episode in enumerate(self.collection["episodes"]):
                source = Path(episode["source"])
                for name, digest in episode["source_sha256"].items():
                    self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
                replay, actions = load_endpoint_replay(source)
                start = episode_index * 450
                stored = values[start:start + 450]
                np.testing.assert_array_equal(np.asarray(stored["action"]), actions)
                np.testing.assert_array_equal(
                    np.asarray(stored["observation.state"]), replay.qpos[0:22500:50, :6].astype(np.float32),
                )
                for frame in (0, 100, 240, 340, 449):
                    item = self.dataset[start + frame]
                    expected = render_observation(renderer, replay.qpos[frame * 50])
                    for key in CAMERAS:
                        actual = item[key].permute(1, 2, 0).numpy()
                        np.testing.assert_allclose(actual, expected[key].astype(np.float32) / 255, atol=1e-7)
        finally:
            renderer.close()

    def test_training_checkpoint_and_contract_rejection(self):
        import torch
        from so101_dice_sim.action_contract import ENDPOINT_MODE, LEGACY_MODE, require_training_contract
        from so101_dice_sim.learning import load_policy

        require_training_contract(self.training, ENDPOINT_MODE)
        with self.assertRaises(ValueError):
            require_training_contract(self.training, LEGACY_MODE)
        self.assertTrue(self.training["monitored_weights_changed"])
        self.assertEqual(self.training["steps"], 30)
        self.assertEqual(
            [row["sample_index"] for row in self.training["losses"]],
            [(step * len(self.dataset) // 30 + 25) % len(self.dataset) for step in range(30)],
        )
        self.assertTrue(all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
                            for row in self.training["losses"]))
        policy, pre, _ = load_policy(self.training["models"], self.dataset, Path(self.training["checkpoint"]))
        batch = pre(torch.utils.data.default_collate([self.dataset[25]]))
        self.assertEqual(tuple(batch["action"].shape), (1, 10, 6))
        self.assertFalse(policy.config.push_to_hub)
        self.assertEqual(self.rollout["action_contract"], self.training["action_contract"])

    def test_every_physics_step_uses_predicted_endpoint_interpolation(self):
        import mujoco
        from so101_dice_sim.learning import bounded_action
        from so101_dice_sim.robot_check import JOINT_NAMES, ROOT

        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
        with (self.rollout_path.parent / "physics.csv").open(encoding="utf-8", newline="") as stream:
            physics = list(csv.DictReader(stream))
        self.assertEqual(self.rollout["controls"], 20)
        self.assertEqual(self.rollout["physics_steps"], 1000)
        self.assertEqual(len(physics), 1000)
        self.assertAlmostEqual(self.rollout["simulated_s"], 2)
        self.assertFalse(self.rollout["task_success_evaluated"])
        targets = np.array([[float(s[f"{name}_target"]) for name in JOINT_NAMES] for s in physics])
        previous = np.array([0., 0, 0, 0, 0, .6])
        self.assertEqual(len(self.rollout["rows"]), 20)
        for i, row in enumerate(self.rollout["rows"]):
            np.testing.assert_array_equal(row["start_action_rad"], previous)
            endpoint = bounded_action(np.array(row["raw_action_rad"]), previous, model.actuator_ctrlrange)
            if row["blocked_open"]:
                endpoint[5] = previous[5]
            np.testing.assert_array_equal(row["applied_action_rad"], endpoint)
            expected = previous + np.arange(1, 51)[:, None] / 50 * (endpoint - previous)
            expected[-1] = endpoint
            np.testing.assert_allclose(targets[i * 50:(i + 1) * 50], expected, atol=1e-14, rtol=0)
            self.assertLessEqual(np.max(np.abs(np.diff(np.vstack([previous, expected]), axis=0))), .001000001)
            self.assertAlmostEqual(row["target_time_s"], (i + 1) / 10)
            self.assertTrue(np.isfinite(row["joint_qpos_rad"]).all())
            previous = endpoint
        for i, row in enumerate(physics):
            self.assertAlmostEqual(float(row["time_s"]), (i + 1) * .002)
            self.assertTrue(all(np.isfinite(float(value)) for value in row.values()))

    @unittest.skipUnless(os.environ.get("ENDPOINT_ABORT"), "3更新モデルの中断記録を明示した場合だけ実行")
    def test_undertrained_contact_is_not_reported_as_success(self):
        path = Path(os.environ["ENDPOINT_ABORT"])
        abort = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(abort["completed"])
        self.assertFalse((path.parent / "rollout.json").exists())
        self.assertTrue((path.parent / "error.json").exists())
        self.assertIn("table", abort["reason"])
        self.assertIn("geom-31", abort["reason"])
        self.assertAlmostEqual(abort["simulated_s"], .75)
        self.assertEqual(abort["completed_controls"], 7)
        self.assertFalse(abort["task_success_evaluated"])
        training = json.loads(Path(abort["training"]).read_text(encoding="utf-8"))
        self.assertEqual(training["steps"], 3)
        self.assertEqual(training["action_contract"], self.training["action_contract"])


if __name__ == "__main__":
    unittest.main()
