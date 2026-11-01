import csv
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("ENDPOINT_REPLAY_SUMMARY"), "補間再生のsummary.jsonを明示した場合だけ実行")
class EndpointReplayIntegrationTest(unittest.TestCase):
    def test_all_physical_steps_and_original_acceptance_checks(self):
        import mujoco
        from so101_dice_sim.grasp import validate_grasp
        from so101_dice_sim.learning import bounded_action
        from so101_dice_sim.placement import validate_placement
        from so101_dice_sim.robot_check import JOINT_NAMES, ROOT
        from so101_dice_sim.rotation import validate_rotation
        from so101_dice_sim.visualization import load_replay

        summary_path = Path(os.environ["ENDPOINT_REPLAY_SUMMARY"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        settings = json.loads((summary_path.parent / "settings.json").read_text(encoding="utf-8"))
        collection_path = Path(settings["collection"])
        self.assertEqual(hashlib.sha256(collection_path.read_bytes()).hexdigest(), settings["collection_sha256"])
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        for name, digest in settings["dataset_sha256"].items():
            path = Path(collection["dataset"]) / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(summary["runs"], 12)
        self.assertEqual(summary["passed_count"], 12)
        self.assertEqual(
            set(settings["modes"]),
            {"source_float32_500hz", "endpoint_linear", "endpoint_linear_bounded", "endpoint_smooth"},
        )
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
        for result in summary["results"]:
            directory = summary_path.parent / result["run_directory"]
            report = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"] and report["completed"] and report["release_executed"])
            self.assertFalse(report["failure_reasons"] or report["warnings"] or report["unexpected_contact_pairs"])
            self.assertEqual(report["physics_steps"], 22500)
            self.assertEqual(report["sample_count"], 22501)
            self.assertAlmostEqual(report["duration_s"], 45)
            self.assertEqual([gate["stage"] for gate in report["gates"]], ["rotation_end", "release"])
            self.assertTrue(all(not gate["failure_reasons"] for gate in report["gates"]))
            source = load_replay(Path(report["source"]))
            episode = collection["episodes"][result["episode"]]
            for name, digest in episode["source_sha256"].items():
                self.assertEqual(hashlib.sha256((Path(report["source"]) / name).read_bytes()).hexdigest(), digest)
            trajectory = directory / "trajectory.csv"
            self.assertEqual(hashlib.sha256(trajectory.read_bytes()).hexdigest(), report["trajectory_sha256"])
            with trajectory.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            samples = [{
                key: value if key == "phase" else value == "True" if value in ("True", "False")
                else None if value == "" else int(value) if key == "top_face" else float(value)
                for key, value in row.items()
            } for row in rows]
            self.assertEqual(validate_grasp(samples[:9501], {}, observe_position=True), report["grasp_validation"])
            self.assertEqual(
                validate_rotation(samples[:12501], {}, target_face=2, observe_position=True),
                report["rotation_validation"],
            )
            self.assertEqual(
                validate_placement(samples, {}, target_face=2, observe_position=True),
                report["placement_validation"],
            )
            self.assertTrue(report["grasp_validation"]["passed"])
            self.assertTrue(report["rotation_validation"]["passed"])
            self.assertTrue(report["placement_validation"]["passed"])
            source_targets = np.array([
                [float(row[f"{name}_target"]) for name in JOINT_NAMES] for row in source.rows
            ])
            actual = np.array([[s[f"{name}_target"] for name in JOINT_NAMES] for s in samples])
            if report["mode"] == "source_float32_500hz":
                self.assertEqual(report["control_updates"], 0)
                np.testing.assert_array_equal(actual[1:], source_targets[1:].astype(np.float32).astype(float))
                continue
            self.assertEqual(report["control_updates"], 450)
            self.assertEqual(len(report["controls"]), 450)
            self.assertIn("t+0.1", report["action_semantics"])
            previous = source_targets[0].copy()
            for index, control in enumerate(report["controls"]):
                step = index * 50
                raw = source_targets[step + 50].astype(np.float32).astype(float)
                np.testing.assert_array_equal(raw, control["raw_action_rad"])
                np.testing.assert_array_equal(previous, control["start_action_rad"])
                self.assertEqual(control["teacher_endpoint_row"], step + 50)
                self.assertAlmostEqual(control["target_time_s"], (step + 50) * .002)
                self.assertAlmostEqual(control["time_s"], step * .002)
                endpoint = (
                    bounded_action(raw, previous, model.actuator_ctrlrange)
                    if report["mode"] == "endpoint_linear_bounded" else raw.copy()
                )
                if control["blocked_open"]:
                    contact = samples[step]
                    self.assertTrue(
                        (contact["fixed_jaw_force_n"] > .01 or contact["moving_jaw_force_n"] > .01)
                        and contact["dice_table_force_n"] <= .01 and endpoint[5] > previous[5]
                    )
                    endpoint[5] = previous[5]
                np.testing.assert_array_equal(endpoint, control["applied_action_rad"])
                fraction = np.arange(1, 51, dtype=float) / 50
                if report["mode"] == "endpoint_smooth":
                    fraction = fraction**2 * (3 - 2 * fraction)
                expected = previous + fraction[:, None] * (endpoint - previous)
                expected[-1] = endpoint
                np.testing.assert_allclose(actual[step + 1:step + 51], expected, atol=1e-14, rtol=0)
                if report["mode"] == "endpoint_linear_bounded":
                    self.assertLessEqual(np.max(np.abs(endpoint - previous)), .0500000001)
                    self.assertLessEqual(
                        np.max(np.abs(np.diff(actual[step:step + 51], axis=0))), .0010000001,
                    )
                previous = endpoint
            self.assertTrue(all(row["unexpected_contacts"] == 0 and row["limits_ok"] for row in samples))


if __name__ == "__main__":
    unittest.main()
