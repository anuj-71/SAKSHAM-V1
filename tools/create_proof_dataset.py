import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.sign_language.dataset_manager import DatasetManager


def make_hand(present: bool, base_x: float, base_y: float) -> dict:
    if not present:
        return {"present": False, "confidence": 0.0, "landmarks": []}

    landmarks = []
    for index in range(21):
        landmarks.append([
            base_x + (index * 0.008),
            base_y + (index * 0.004),
            0.0,
        ])
    return {"present": True, "confidence": 0.95, "landmarks": landmarks}


def add_clip(manager: DatasetManager, label: str, signer_id: str, left_present: bool, right_present: bool, frame_count: int):
    manager.start_recording(label, signer_id)
    for frame_index in range(frame_count):
        hand_data = {
            "left_hand": make_hand(left_present, 0.15 + (frame_index * 0.001), 0.20),
            "right_hand": make_hand(right_present, 0.60 - (frame_index * 0.001), 0.25),
        }
        manager.add_frame_data(hand_data)

    review = manager.stop_recording()
    if not review or not review["passes_quality_checks"]:
        raise RuntimeError(f"Proof clip failed quality checks for {label}/{signer_id}: {review}")

    accepted = manager.accept_current_clip()
    if not accepted or accepted["status"] != "accepted":
        raise RuntimeError(f"Proof clip was not accepted for {label}/{signer_id}: {accepted}")
    return accepted


def main():
    dataset_dir = os.path.join(REPO_ROOT, "dataset")
    export_dir = os.path.join(REPO_ROOT, "exports")
    manager = DatasetManager(data_dir=dataset_dir, export_dir=export_dir)

    clips = [
        ("HELLO", "SIGNER_01", False, True, 24),
        ("HELP", "SIGNER_02", True, False, 22),
        ("NO_SIGN", "SIGNER_03", True, True, 20),
    ]

    accepted_samples = []
    for label, signer_id, left_present, right_present, frame_count in clips:
        accepted_samples.append(add_clip(manager, label, signer_id, left_present, right_present, frame_count))

    exported_files = manager.export_dataset("both")

    print("Created proof dataset samples:")
    for sample in accepted_samples:
        print(f"- {sample['sample_id']} ({sample['label']} / {sample['signer_id']})")

    print("Exported summary files:")
    for path in exported_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
