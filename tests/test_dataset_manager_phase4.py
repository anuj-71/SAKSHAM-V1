import json
import os
import tempfile
import unittest

from src.sign_language.dataset_manager import DatasetManager


def make_hand(present: bool, base_x: float) -> dict:
    landmarks = []
    for index in range(21):
        landmarks.append([base_x + (index * 0.01), 0.2 + (index * 0.005), 0.0])
    return {
        "present": present,
        "confidence": 0.95 if present else 0.0,
        "landmarks": landmarks if present else [],
    }


def make_frame(left_present: bool, right_present: bool) -> dict:
    return {
        "left_hand": make_hand(left_present, 0.1),
        "right_hand": make_hand(right_present, 0.6),
    }


class TestDatasetManagerPhase4(unittest.TestCase):
    def test_accept_creates_npz_manifest_and_exports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "dataset")
            export_dir = os.path.join(temp_dir, "exports")
            manager = DatasetManager(data_dir=dataset_dir, export_dir=export_dir)

            manager.start_recording("HELLO", "SIGNER_01")
            for _ in range(20):
                manager.add_frame_data(make_frame(left_present=False, right_present=True))

            review = manager.stop_recording()
            self.assertTrue(review["passes_quality_checks"])
            self.assertEqual(review["frame_count"], 20)

            accepted = manager.accept_current_clip()
            self.assertEqual(accepted["status"], "accepted")
            self.assertTrue(os.path.exists(manager.manifest_path))
            self.assertTrue(os.path.exists(os.path.join(manager.accepted_dir, f"{accepted['sample_id']}.npz")))
            self.assertTrue(os.path.exists(os.path.join(manager.accepted_dir, f"{accepted['sample_id']}.json")))

            with open(manager.manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest_entries = [json.loads(line) for line in manifest_file if line.strip()]
            self.assertEqual(len(manifest_entries), 1)
            self.assertEqual(manifest_entries[0]["sample_id"], accepted["sample_id"])

            exported_files = manager.export_dataset("both")
            self.assertEqual(len(exported_files), 2)
            for export_path in exported_files:
                self.assertTrue(os.path.exists(export_path))

    def test_short_clip_is_blocked_from_acceptance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(
                data_dir=os.path.join(temp_dir, "dataset"),
                export_dir=os.path.join(temp_dir, "exports"),
            )

            manager.start_recording("HELP", "SIGNER_02")
            for _ in range(5):
                manager.add_frame_data(make_frame(left_present=True, right_present=False))

            review = manager.stop_recording()
            self.assertFalse(review["passes_quality_checks"])
            self.assertTrue(review["quality_blockers"])

            blocked = manager.accept_current_clip()
            self.assertEqual(blocked["status"], "blocked")
            self.assertFalse(os.path.exists(manager.manifest_path))
            self.assertTrue(manager.has_review_clip())

    def test_statistics_audit_and_two_hand_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(
                data_dir=os.path.join(temp_dir, "dataset"),
                export_dir=os.path.join(temp_dir, "exports"),
            )

            manager.start_recording("HELLO", "SIGNER_01")
            for _ in range(24):
                manager.add_frame_data(make_frame(left_present=True, right_present=True))
            manager.stop_recording()
            manager.accept_current_clip()

            manager.start_recording("HELP", "SIGNER_02")
            for _ in range(20):
                manager.add_frame_data(make_frame(left_present=False, right_present=True))
            manager.stop_recording()
            manager.accept_current_clip()

            report = manager.build_collection_report(["HELLO", "HELP", "WATER"])
            statistics = report["statistics"]
            audit = report["audit"]
            two_hand = report["two_hand_verification"]

            self.assertEqual(statistics["total_clips"], 2)
            self.assertEqual(statistics["clips_per_label"]["HELLO"], 1)
            self.assertEqual(statistics["clips_per_label"]["HELP"], 1)
            self.assertEqual(statistics["clips_per_label"]["WATER"], 0)
            self.assertEqual(statistics["clips_per_signer"]["SIGNER_01"], 1)
            self.assertEqual(statistics["clips_per_signer"]["SIGNER_02"], 1)
            self.assertGreater(statistics["average_frame_count"], 20.0)
            self.assertGreater(statistics["hand_visibility_metrics"]["both_hands_avg"], 0.0)
            self.assertIn("WATER", statistics["class_balance"]["missing_labels"])

            self.assertEqual(audit["verified_samples"], 2)
            self.assertEqual(len(audit["issues"]), 0)
            self.assertEqual(audit["samples_with_both_hands"], 1)

            self.assertTrue(two_hand["two_hand_capture_verified"])
            self.assertEqual(two_hand["samples_with_both_hands"], 1)
            self.assertTrue(two_hand["verified_sample_id"].startswith("HELLO_"))

    def test_live_recording_feedback_tracks_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(
                data_dir=os.path.join(temp_dir, "dataset"),
                export_dir=os.path.join(temp_dir, "exports"),
            )

            manager.start_recording("YES", "SIGNER_03")
            for _ in range(5):
                manager.add_frame_data(make_frame(left_present=True, right_present=False))

            feedback = manager.get_live_recording_feedback()
            self.assertIsNotNone(feedback)
            self.assertEqual(feedback["frame_count"], 5)
            self.assertEqual(feedback["frames_remaining_to_minimum"], 10)
            self.assertTrue(feedback["left_visible_now"])
            self.assertFalse(feedback["right_visible_now"])
            self.assertEqual(feedback["quality_state"], "needs_frames")
            self.assertIn("Collect 10 more frames", feedback["feedback"])


if __name__ == "__main__":
    unittest.main()
