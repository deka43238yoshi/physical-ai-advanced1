import csv
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("TEACHER_REPLAY_SUMMARY"), "教師再生のsummary.jsonを明示した場合だけ実行")
class TeacherReplayIntegrationTest(unittest.TestCase):
    def test_saved_study_against_sources_and_physical_evaluators(self):
        from so101_dice_sim.grasp import validate_grasp
        from so101_dice_sim.learning import bounded_action
        from so101_dice_sim.placement import validate_placement
        from so101_dice_sim.robot_check import JOINT_NAMES, ROOT
        from so101_dice_sim.rotation import validate_rotation
        from so101_dice_sim.visualization import load_replay
        import mujoco

        summary_path = Path(os.environ["TEACHER_REPLAY_SUMMARY"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        settings = json.loads((summary_path.parent / "settings.json").read_text(encoding="utf-8"))
        collection_path = Path(settings["collection"])
        self.assertEqual(hashlib.sha256(collection_path.read_bytes()).hexdigest(), settings["collection_sha256"])
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        for name, digest in settings["dataset_sha256"].items():
            path = Path(collection["dataset"]) / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(summary["runs"], 9)
        self.assertEqual(summary["passed_count"], 3)
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "so101_dice_scene.xml"))
        for result in summary["results"]:
            directory = summary_path.parent / result["run_directory"]
            report = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
            source = load_replay(Path(report["source"]))
            trajectory = directory / ("trajectory.csv" if report["completed"] else "aborted_trajectory.csv")
            with trajectory.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            samples = [{
                key: value if key == "phase" else value == "True" if value in ("True", "False")
                else None if value == "" else int(value) if key == "top_face" else float(value)
                for key, value in row.items()
            } for row in rows]
            self.assertEqual(report["sample_count"], len(samples))
            self.assertEqual(report["physics_steps"], len(samples) - 1)
            self.assertEqual(report["passed"], result["passed"])
            self.assertEqual(hashlib.sha256(trajectory.read_bytes()).hexdigest(), report["trajectory_sha256"])
            self.assertFalse(report["warnings"])
            self.assertTrue(all(s["unexpected_contacts"] == 0 and s["limits_ok"] for s in samples))
            self.assertTrue(validate_grasp(samples[:9501], {}, observe_position=True)["passed"])
            rotation = validate_rotation(samples[:12501], {}, target_face=2, observe_position=True)
            self.assertEqual(rotation, report["rotation_validation"])
            if report["mode"] == "source_500hz":
                self.assertTrue(report["passed"] and report["completed"] and report["release_executed"])
                self.assertEqual(trajectory.read_bytes(), (Path(report["source"]) / "trajectory.csv").read_bytes())
                self.assertLess(report["max_qpos_deviation_from_source"], 1e-12)
                self.assertTrue(validate_placement(samples, {}, target_face=2, observe_position=True)["passed"])
                continue
            self.assertFalse(report["passed"] or report["completed"] or report["release_executed"])
            self.assertEqual(report["stopped_before_phase"], "place_above")
            self.assertEqual(report["physics_steps"], 12500)
            self.assertEqual(report["failure_reasons"], ["rotation.no_position_slip", "rotation.hold_speeds"])
            self.assertIsNone(report["placement_validation"])
            self.assertTrue(rotation["checks"]["requested_face_up"])
            self.assertGreaterEqual(rotation["max_relative_drift_m"], .002)
            self.assertEqual(len(report["controls"]), 250)
            previous = np.array([float(source.rows[0][f"{name}_target"]) for name in JOINT_NAMES])
            targets = np.array([[s[f"{name}_target"] for name in JOINT_NAMES] for s in samples])
            for index, control in enumerate(report["controls"]):
                step = index * 50
                raw = np.array([
                    float(source.rows[step + 1][f"{name}_target"]) for name in JOINT_NAMES
                ], dtype=np.float32).astype(float)
                np.testing.assert_array_equal(raw, control["raw_action_rad"])
                applied = (
                    bounded_action(raw, previous, model.actuator_ctrlrange)
                    if report["mode"] == "dataset_10hz_bounded" else raw
                )
                if control["blocked_open"]:
                    contact = samples[step]
                    self.assertTrue(
                        (contact["fixed_jaw_force_n"] > .01 or contact["moving_jaw_force_n"] > .01)
                        and contact["dice_table_force_n"] <= .01 and applied[5] > previous[5]
                    )
                    applied[5] = previous[5]
                np.testing.assert_array_equal(applied, control["applied_action_rad"])
                np.testing.assert_array_equal(targets[step + 1:step + 51], np.tile(applied, (50, 1)))
                self.assertEqual(control["safety_adjusted"], not np.array_equal(raw, applied))
                previous = applied


if __name__ == "__main__":
    unittest.main()
