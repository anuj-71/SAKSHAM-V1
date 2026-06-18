import os
import csv
import json
import time
import logging
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetManager:
    """
    Manages the collection, storage, and export of sign language dataset sequences.
    Records temporal sequences of hand landmarks for future dynamic sign language model training.
    """
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.accepted_dir = os.path.join(self.data_dir, "accepted")
        self.rejected_dir = os.path.join(self.data_dir, "rejected")
        os.makedirs(self.accepted_dir, exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)
            
        self.is_recording = False
        self.current_label = ""
        self.current_signer_id = ""
        self.current_sequence_id = ""
        self.current_sequence = []
        self.review_clip: Optional[Dict] = None
        self.manifest_path = os.path.join(self.data_dir, "manifest.jsonl")
        
        # Meta storage for multiple sequences
        self.session_sequences = []

    def start_recording(self, label: str, signer_id: str = "SIGNER_01"):
        """Starts recording a new temporal sequence of landmarks for the given sign label."""
        if self.is_recording:
            self.stop_recording()
            
        self.current_label = label.upper()
        self.current_signer_id = signer_id
        self.current_sequence_id = f"{self.current_label}_{int(time.time() * 1000)}"
        self.current_sequence = []
        self.is_recording = True
        self.review_clip = None
        logging.info(f"DatasetManager: Started recording sequence for '{self.current_label}' ({self.current_signer_id})")

    @staticmethod
    def _hand_entry(hand: Dict) -> Dict:
        if not hand or not hand.get("present"):
            return {
                "present": False,
                "confidence": 0.0,
                "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)],
            }
        return {
            "present": True,
            "confidence": float(hand.get("confidence", 0.0)),
            "landmarks": [list(pt) for pt in hand.get("landmarks", [])],
        }

    def add_frame_data(self, hand_data: Dict):
        """Adds a single frame's hand landmark data to the current sequence."""
        if not self.is_recording or not hand_data:
            return

        left_hand = self._hand_entry(hand_data.get("left_hand", {}))
        right_hand = self._hand_entry(hand_data.get("right_hand", {}))
        frame_entry = {
            "timestamp": time.time(),
            "sequence_id": self.current_sequence_id,
            "label": self.current_label,
            "signer_id": self.current_signer_id,
            "left_hand": left_hand,
            "right_hand": right_hand,
        }
        self.current_sequence.append(frame_entry)

    def stop_recording(self) -> Optional[Dict]:
        """Stops the recording and prepares the clip for accept/reject review."""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if len(self.current_sequence) > 0:
            review_clip = {
                "sequence_id": self.current_sequence_id,
                "label": self.current_label,
                "signer_id": self.current_signer_id,
                "frames": self.current_sequence
            }
            self.review_clip = review_clip
            logging.info(f"DatasetManager: Stopped recording. Sequence '{self.current_sequence_id}' has {len(self.current_sequence)} frames.")
            self.current_sequence = []
            return self.get_review_summary()
        self.current_sequence = []
        return None

    def has_review_clip(self) -> bool:
        return self.review_clip is not None

    def get_review_summary(self) -> Optional[Dict]:
        if not self.review_clip:
            return None

        frames = self.review_clip["frames"]
        frame_count = len(frames)
        left_present = sum(1 for frame in frames if frame["left_hand"]["present"])
        right_present = sum(1 for frame in frames if frame["right_hand"]["present"])
        summary = {
            "sequence_id": self.review_clip["sequence_id"],
            "label": self.review_clip["label"],
            "signer_id": self.review_clip["signer_id"],
            "frame_count": frame_count,
            "left_presence_ratio": (left_present / frame_count) if frame_count else 0.0,
            "right_presence_ratio": (right_present / frame_count) if frame_count else 0.0,
        }
        return summary

    def _save_clip_metadata(self, target_dir: str, metadata: Dict):
        metadata_path = os.path.join(target_dir, f"{metadata['sample_id']}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def _save_clip_arrays(self, target_dir: str, sample_id: str, frames: List[Dict]):
        timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float32)
        left_hand = np.asarray([frame["left_hand"]["landmarks"] for frame in frames], dtype=np.float32)
        right_hand = np.asarray([frame["right_hand"]["landmarks"] for frame in frames], dtype=np.float32)
        left_present = np.asarray([float(frame["left_hand"]["present"]) for frame in frames], dtype=np.float32)
        right_present = np.asarray([float(frame["right_hand"]["present"]) for frame in frames], dtype=np.float32)
        left_confidence = np.asarray([frame["left_hand"]["confidence"] for frame in frames], dtype=np.float32)
        right_confidence = np.asarray([frame["right_hand"]["confidence"] for frame in frames], dtype=np.float32)
        np.savez_compressed(
            os.path.join(target_dir, f"{sample_id}.npz"),
            timestamps=timestamps,
            left_hand=left_hand,
            right_hand=right_hand,
            left_present=left_present,
            right_present=right_present,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
        )

    def accept_current_clip(self) -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = self.get_review_summary()
        sample_id = self.review_clip["sequence_id"]
        metadata = {
            "sample_id": sample_id,
            "label": self.review_clip["label"],
            "signer_id": self.review_clip["signer_id"],
            "frame_count": summary["frame_count"],
            "left_presence_ratio": summary["left_presence_ratio"],
            "right_presence_ratio": summary["right_presence_ratio"],
            "saved_at": time.time(),
            "status": "accepted",
        }

        self._save_clip_arrays(self.accepted_dir, sample_id, self.review_clip["frames"])
        self._save_clip_metadata(self.accepted_dir, metadata)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

        self.session_sequences.append(metadata)
        self.review_clip = None
        return metadata

    def reject_current_clip(self, reason: str = "Rejected") -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = self.get_review_summary()
        sample_id = self.review_clip["sequence_id"]
        metadata = {
            "sample_id": sample_id,
            "label": self.review_clip["label"],
            "signer_id": self.review_clip["signer_id"],
            "frame_count": summary["frame_count"],
            "left_presence_ratio": summary["left_presence_ratio"],
            "right_presence_ratio": summary["right_presence_ratio"],
            "saved_at": time.time(),
            "status": "rejected",
            "reason": reason,
        }
        self._save_clip_metadata(self.rejected_dir, metadata)
        self.review_clip = None
        return metadata

    def export_dataset(self, format: str = "json"):
        """Exports all recorded sequences in the session to disk."""
        if not self.session_sequences:
            logging.info("DatasetManager: No sequences to export.")
            return
            
        timestamp = int(time.time())
        if format == "json":
            file_path = os.path.join(self.data_dir, f"dataset_export_{timestamp}.json")
            with open(file_path, "w") as f:
                json.dump(self.session_sequences, f, indent=4)
            logging.info(f"DatasetManager: Exported {len(self.session_sequences)} sequences to {file_path}")
            
        elif format == "csv":
            file_path = os.path.join(self.data_dir, f"dataset_export_{timestamp}.csv")
            with open(file_path, "w", newline='') as f:
                writer = csv.writer(f)
                header = [
                    "sample_id",
                    "label",
                    "signer_id",
                    "frame_count",
                    "left_presence_ratio",
                    "right_presence_ratio",
                    "status",
                ]
                writer.writerow(header)
                
                for seq in self.session_sequences:
                    writer.writerow([
                        seq["sample_id"],
                        seq["label"],
                        seq["signer_id"],
                        seq["frame_count"],
                        seq["left_presence_ratio"],
                        seq["right_presence_ratio"],
                        seq["status"],
                    ])
            logging.info(f"DatasetManager: Exported {len(self.session_sequences)} sequences to {file_path}")
