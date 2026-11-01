import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(
    all(os.environ.get(key) for key in ("LEARNING_COLLECTION", "LEARNING_TRAINING", "LEARNING_ROLLOUT")),
    "ローカル実証の3つのJSONを環境変数で明示した場合だけ実行",
)
class LearningIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        cls.collection = json.loads(Path(os.environ["LEARNING_COLLECTION"]).read_text(encoding="utf-8"))
        cls.training = json.loads(Path(os.environ["LEARNING_TRAINING"]).read_text(encoding="utf-8"))
        cls.rollout = json.loads(Path(os.environ["LEARNING_ROLLOUT"]).read_text(encoding="utf-8"))
        cls.dataset = LeRobotDataset(
            "local/so101_face2_smoke", root=Path(cls.collection["dataset"]),
            delta_timestamps={"action": [i / 10 for i in range(10)]},
        )

    def test_all_teacher_states_and_next_step_commands(self):
        from so101_dice_sim.robot_check import JOINT_NAMES
        from so101_dice_sim.learning import CAMERAS, teacher_indices
        from so101_dice_sim.visualization import load_replay

        self.assertEqual(self.dataset.meta.total_episodes, 3)
        self.assertEqual(len(self.dataset), 1350)
        self.assertEqual(
            {key for key in self.dataset.features if key.startswith("observation.")},
            {"observation.state", *CAMERAS},
        )
        values = self.dataset.hf_dataset.select_columns(["observation.state", "action"])
        offset = 0
        for episode in self.collection["episodes"]:
            source = Path(episode["source"])
            for name, digest in episode["source_sha256"].items():
                self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
            replay = load_replay(source)
            indices = teacher_indices(len(replay.rows), replay.report["timestep_s"])
            self.assertEqual(len(indices), episode["frames"])
            actual_states = np.asarray(values[offset:offset + len(indices)]["observation.state"])
            actual_actions = np.asarray(values[offset:offset + len(indices)]["action"])
            np.testing.assert_allclose(actual_states, replay.qpos[list(indices), :6], atol=1e-7)
            expected_actions = [
                [float(replay.rows[i + 1][f"{name}_target"]) for name in JOINT_NAMES]
                for i in indices
            ]
            np.testing.assert_allclose(actual_actions, expected_actions, atol=1e-7)
            offset += len(indices)
        self.assertEqual(offset, len(self.dataset))

    def test_saved_policy_strict_load_and_training_batch_shapes(self):
        import torch
        from so101_dice_sim.learning import CAMERAS, load_policy

        policy, pre, _ = load_policy(self.training["models"], self.dataset, Path(self.training["checkpoint"]))
        batch = pre(torch.utils.data.default_collate([self.dataset[25]]))
        self.assertEqual(tuple(batch["observation.state"].shape), (1, 6))
        self.assertEqual(tuple(batch["action"].shape), (1, 10, 6))
        self.assertEqual(tuple(batch["action_is_pad"].shape), (1, 10))
        for key in CAMERAS:
            self.assertEqual(tuple(batch[key].shape), (1, 3, 256, 256))
            self.assertTrue(torch.isfinite(batch[key]).all())
        self.assertFalse(policy.config.push_to_hub)
        self.assertTrue(self.training["monitored_weights_changed"])
        self.assertEqual(self.training["steps"], 3)
        self.assertEqual([row["sample_index"] for row in self.training["losses"]], [25, 475, 925])
        self.assertTrue(self.training["pretrained_weights_strict"])
        self.assertTrue(all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
                            for row in self.training["losses"]))

    def test_physical_rollout_commands_and_limits(self):
        from so101_dice_sim.learning import bounded_action
        from so101_dice_sim.visualization import ReplayRenderer

        renderer = ReplayRenderer()
        try:
            model = renderer.model
            previous = np.zeros(6)
            previous[5] = .6
            self.assertEqual(self.rollout["controls"], 20)
            self.assertEqual(self.rollout["physics_steps"], 1000)
            self.assertAlmostEqual(self.rollout["simulated_s"], 2.)
            self.assertFalse(self.rollout["task_success_evaluated"])
            self.assertFalse(self.rollout["uploaded"])
            self.assertEqual(len(self.rollout["rows"]), 20)
            for index, row in enumerate(self.rollout["rows"]):
                applied = np.array(row["applied_action_rad"])
                expected = bounded_action(np.array(row["raw_action_rad"]), previous, model.actuator_ctrlrange)
                if row["blocked_open"]:
                    expected[5] = previous[5]
                np.testing.assert_allclose(applied, expected)
                self.assertTrue(np.all(np.abs(applied - previous) <= .05000001))
                self.assertTrue(np.isfinite(row["joint_qpos_rad"]).all())
                self.assertAlmostEqual(row["time_s"], (index + 1) / 10)
                previous = applied
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
