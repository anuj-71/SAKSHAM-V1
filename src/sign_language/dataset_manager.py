import os
import csv
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetManager:
    """
    Manages the collection, storage, and export of sign language dataset sequences.
    Records temporal sequences of hand landmarks for future dynamic sign language model training.
    """
    MIN_FRAME_COUNT = 15
    RECOMMENDED_FRAME_COUNT = 24
    MIN_ACTIVE_HAND_PRESENCE_RATIO = 0.60
    RECOMMENDED_ACTIVE_HAND_PRESENCE_RATIO = 0.85

    def __init__(self, data_dir: str = "dataset", export_dir: str = "exports"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

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
                "frames": self.current_sequence,
            }
            self.review_clip = review_clip
            logging.info(f"DatasetManager: Stopped recording. Sequence '{self.current_sequence_id}' has {len(self.current_sequence)} frames.")
            self.current_sequence = []
            return self.get_review_summary()
        self.current_sequence = []
        return None

    def has_review_clip(self) -> bool:
        return self.review_clip is not None

    def _summarize_frames(self, frames: List[Dict]) -> Dict:
        frame_count = len(frames)
        left_present = sum(1 for frame in frames if frame["left_hand"]["present"])
        right_present = sum(1 for frame in frames if frame["right_hand"]["present"])
        both_hands_present = sum(
            1 for frame in frames if frame["left_hand"]["present"] and frame["right_hand"]["present"]
        )
        left_presence_ratio = (left_present / frame_count) if frame_count else 0.0
        right_presence_ratio = (right_present / frame_count) if frame_count else 0.0
        active_hand_ratio = max(left_presence_ratio, right_presence_ratio)
        both_hands_ratio = (both_hands_present / frame_count) if frame_count else 0.0

        blocking_issues = []
        warnings = []
        if frame_count < self.MIN_FRAME_COUNT:
            blocking_issues.append(
                f"Need at least {self.MIN_FRAME_COUNT} frames; recorded {frame_count}."
            )
        elif frame_count < self.RECOMMENDED_FRAME_COUNT:
            warnings.append(
                f"Short clip: {frame_count} frames. Recommended at least {self.RECOMMENDED_FRAME_COUNT}."
            )

        if active_hand_ratio < self.MIN_ACTIVE_HAND_PRESENCE_RATIO:
            blocking_issues.append(
                f"Low hand visibility: best hand tracked in {active_hand_ratio:.0%} of frames; minimum is {self.MIN_ACTIVE_HAND_PRESENCE_RATIO:.0%}."
            )
        elif active_hand_ratio < self.RECOMMENDED_ACTIVE_HAND_PRESENCE_RATIO:
            warnings.append(
                f"Hand visibility is only {active_hand_ratio:.0%}. Recommended at least {self.RECOMMENDED_ACTIVE_HAND_PRESENCE_RATIO:.0%}."
            )

        return {
            "frame_count": frame_count,
            "left_presence_ratio": left_presence_ratio,
            "right_presence_ratio": right_presence_ratio,
            "active_hand_ratio": active_hand_ratio,
            "both_hands_ratio": both_hands_ratio,
            "quality_blockers": blocking_issues,
            "quality_warnings": warnings,
            "passes_quality_checks": len(blocking_issues) == 0,
        }

    def get_review_summary(self) -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = {
            "sequence_id": self.review_clip["sequence_id"],
            "label": self.review_clip["label"],
            "signer_id": self.review_clip["signer_id"],
        }
        summary.update(self._summarize_frames(self.review_clip["frames"]))
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

    def _build_metadata(self, summary: Dict, status: str, reason: Optional[str] = None) -> Dict:
        metadata = {
            "sample_id": summary["sequence_id"],
            "label": summary["label"],
            "signer_id": summary["signer_id"],
            "frame_count": summary["frame_count"],
            "left_presence_ratio": summary["left_presence_ratio"],
            "right_presence_ratio": summary["right_presence_ratio"],
            "active_hand_ratio": summary["active_hand_ratio"],
            "both_hands_ratio": summary["both_hands_ratio"],
            "quality_warnings": summary["quality_warnings"],
            "quality_blockers": summary["quality_blockers"],
            "passes_quality_checks": summary["passes_quality_checks"],
            "saved_at": time.time(),
            "status": status,
        }
        if reason:
            metadata["reason"] = reason
        return metadata

    def accept_current_clip(self) -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = self.get_review_summary()
        if not summary["passes_quality_checks"]:
            blocked = self._build_metadata(summary, status="blocked")
            blocked["message"] = "Clip does not meet dataset quality thresholds."
            return blocked

        sample_id = self.review_clip["sequence_id"]
        metadata = self._build_metadata(summary, status="accepted")
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
        metadata = self._build_metadata(summary, status="rejected", reason=reason)
        self._save_clip_metadata(self.rejected_dir, metadata)
        self.review_clip = None
        return metadata

    def _load_manifest_entries(self) -> List[Dict]:
        if not os.path.exists(self.manifest_path):
            return []

        entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def export_dataset(self, format: str = "both") -> List[str]:
        """Exports accepted dataset metadata to disk and returns the created file paths."""
        manifest_entries = self._load_manifest_entries()
        records = manifest_entries if manifest_entries else list(self.session_sequences)
        if not records:
            logging.info("DatasetManager: No accepted sequences to export.")
            return []

        export_formats = ("json", "csv") if format == "both" else (format,)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_base = os.path.join(self.export_dir, f"dataset_export_{timestamp}")
        created_files: List[str] = []

        if "json" in export_formats:
            json_path = f"{export_base}.json"
            payload = {
                "generated_at": datetime.now().isoformat(),
                "sample_count": len(records),
                "source": "manifest" if manifest_entries else "session",
                "samples": records,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            created_files.append(json_path)

        if "csv" in export_formats:
            csv_path = f"{export_base}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    "sample_id",
                    "label",
                    "signer_id",
                    "frame_count",
                    "left_presence_ratio",
                    "right_presence_ratio",
                    "active_hand_ratio",
                    "both_hands_ratio",
                    "status",
                    "passes_quality_checks",
                    "quality_warnings",
                ]
                writer.writerow(header)

                for seq in records:
                    writer.writerow([
                        seq.get("sample_id", ""),
                        seq.get("label", ""),
                        seq.get("signer_id", ""),
                        seq.get("frame_count", 0),
                        seq.get("left_presence_ratio", 0.0),
                        seq.get("right_presence_ratio", 0.0),
                        seq.get("active_hand_ratio", 0.0),
                        seq.get("both_hands_ratio", 0.0),
                        seq.get("status", ""),
                        seq.get("passes_quality_checks", True),
                        " | ".join(seq.get("quality_warnings", [])),
                    ])
            created_files.append(csv_path)

        logging.info(f"DatasetManager: Exported {len(records)} accepted sequences to {', '.join(created_files)}")
        return created_files
