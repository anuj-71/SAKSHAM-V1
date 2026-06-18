import os
import tempfile
import unittest

from src.sign_language.dataset_manager import DatasetManager
from src.sign_language.training.loader import load_manifest, load_sample
from src.sign_language.training.preprocessing import preprocess_sample_for_gru
from src.sign_language.training.splits import split_records_by_signer


def make_hand(present: bool, base_x: float) -> dict:
    landmarks = []
    for index in range(21):
        landmarks.append([base_x + (index * 0.01), 0.15 + (index * 0.004), 0.0])
    return {
        "present": present,
        "confidence": 0.9 if present else 0.0,
        "landmarks": landmarks if present else [],
    }


def make_frame(left_present: bool, right_present: bool) -> dict:
    return {
        "left_hand": make_hand(left_present, 0.2),
        "right_hand": make_hand(right_present, 0.7),
    }


class TestTrainingPipeline(unittest.TestCase):
    def test_signer_split_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "dataset")
            export_dir = os.path.join(temp_dir, "exports")
            manager = DatasetManager(data_dir=dataset_dir, export_dir=export_dir)

            for signer_id in ("SIGNER_01", "SIGNER_02", "SIGNER_03"):
                manager.start_recording("HELLO", signer_id)
                for _ in range(18):
                    manager.add_frame_data(make_frame(left_present=True, right_present=True))
                manager.stop_recording()
                accepted = manager.accept_current_clip()
                self.assertEqual(accepted["status"], "accepted")

            records = load_manifest(manager.manifest_path)
            self.assertEqual(len(records), 3)

            split_map = split_records_by_signer(records, seed=7)
            train_signers = {record.signer_id for record in split_map["train"]}
            val_signers = {record.signer_id for record in split_map["val"]}
            test_signers = {record.signer_id for record in split_map["test"]}

            self.assertFalse(train_signers & val_signers)
            self.assertFalse(train_signers & test_signers)
            self.assertFalse(val_signers & test_signers)

            sample = load_sample(records[0])
            processed = preprocess_sample_for_gru(sample, target_length=24)
            self.assertEqual(processed["features"].shape, (24, 128))
            self.assertEqual(processed["sequence_mask"].shape, (24,))
            self.assertEqual(processed["stacked_hands"].shape, (18, 2, 21, 3))


if __name__ == "__main__":
    unittest.main()
