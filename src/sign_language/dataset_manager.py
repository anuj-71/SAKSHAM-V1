import os
import csv
import json
import time
import logging
from typing import List, Dict, Optional

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
            
        self.is_recording = False
        self.current_label = ""
        self.current_sequence_id = ""
        self.current_sequence = []
        
        # Meta storage for multiple sequences
        self.session_sequences = []

    def start_recording(self, label: str):
        """Starts recording a new temporal sequence of landmarks for the given sign label."""
        if self.is_recording:
            self.stop_recording()
            
        self.current_label = label.upper()
        self.current_sequence_id = f"{self.current_label}_{int(time.time() * 1000)}"
        self.current_sequence = []
        self.is_recording = True
        logging.info(f"DatasetManager: Started recording sequence for '{self.current_label}'")

    def add_frame_data(self, hand_data: Dict):
        """Adds a single frame's hand landmark data to the current sequence."""
        if not self.is_recording or not hand_data:
            return
            
        # Extract features for storage
        frame_entry = {
            "timestamp": time.time(),
            "sequence_id": self.current_sequence_id,
            "label": self.current_label,
            "hand_label": hand_data.get("label", "Unknown"),
            "confidence": hand_data.get("confidence", 0.0),
            # Flatten normalized landmarks (21 points * 3 dims = 63 features)
            "landmarks": [coord for pt in hand_data.get("landmarks", []) for coord in pt]
        }
        self.current_sequence.append(frame_entry)

    def stop_recording(self):
        """Stops the recording and saves the sequence to the session buffer."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if len(self.current_sequence) > 0:
            self.session_sequences.append({
                "sequence_id": self.current_sequence_id,
                "label": self.current_label,
                "frames": self.current_sequence
            })
            logging.info(f"DatasetManager: Stopped recording. Sequence '{self.current_sequence_id}' has {len(self.current_sequence)} frames.")
        self.current_sequence = []

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
                # Header: meta + 63 landmark coordinates
                header = ["timestamp", "sequence_id", "label", "hand_label", "confidence"]
                for i in range(21):
                    header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
                writer.writerow(header)
                
                for seq in self.session_sequences:
                    for frame in seq["frames"]:
                        row = [
                            frame["timestamp"],
                            frame["sequence_id"],
                            frame["label"],
                            frame["hand_label"],
                            frame["confidence"]
                        ]
                        row.extend(frame["landmarks"])
                        writer.writerow(row)
            logging.info(f"DatasetManager: Exported {len(self.session_sequences)} sequences to {file_path}")
