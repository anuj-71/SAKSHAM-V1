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


if __name__ == "__main__":
    unittest.main()
