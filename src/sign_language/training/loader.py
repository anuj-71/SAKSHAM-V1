import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    label: str
    signer_id: str
    frame_count: int
    metadata: Dict
    npz_path: str
    metadata_path: str


def load_manifest(manifest_path: str, accepted_dir: Optional[str] = None) -> List[DatasetRecord]:
    """Loads accepted sample metadata from a JSONL manifest."""
    if not os.path.exists(manifest_path):
        return []

    resolved_accepted_dir = accepted_dir or os.path.join(os.path.dirname(manifest_path), "accepted")
    records: List[DatasetRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metadata = json.loads(line)
            sample_id = metadata["sample_id"]
            records.append(
                DatasetRecord(
                    sample_id=sample_id,
                    label=metadata["label"],
                    signer_id=metadata["signer_id"],
                    frame_count=int(metadata["frame_count"]),
                    metadata=metadata,
                    npz_path=os.path.join(resolved_accepted_dir, f"{sample_id}.npz"),
                    metadata_path=os.path.join(resolved_accepted_dir, f"{sample_id}.json"),
                )
            )
    return records


def load_sample(record: DatasetRecord) -> Dict[str, np.ndarray]:
    """Loads a single accepted sample bundle from disk."""
    if not os.path.exists(record.npz_path):
        raise FileNotFoundError(f"Accepted sample not found: {record.npz_path}")

    with np.load(record.npz_path) as data:
        sample = {key: data[key] for key in data.files}
    sample["metadata"] = record.metadata
    return sample


def load_dataset(manifest_path: str, accepted_dir: Optional[str] = None) -> List[Dict]:
    """Loads all accepted samples referenced by the manifest."""
    records = load_manifest(manifest_path, accepted_dir=accepted_dir)
    return [load_sample(record) for record in records]
