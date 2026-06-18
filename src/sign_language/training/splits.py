import random
from typing import Dict, List

from src.sign_language.training.loader import DatasetRecord


def _allocate_signer_counts(total_signers: int) -> Dict[str, int]:
    if total_signers <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if total_signers == 1:
        return {"train": 1, "val": 0, "test": 0}
    if total_signers == 2:
        return {"train": 1, "val": 0, "test": 1}

    train_count = max(1, int(round(total_signers * 0.7)))
    val_count = max(1, int(round(total_signers * 0.15)))
    test_count = total_signers - train_count - val_count

    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        else:
            val_count = max(1, val_count - 1)

    while train_count + val_count + test_count > total_signers:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    while train_count + val_count + test_count < total_signers:
        train_count += 1

    return {"train": train_count, "val": val_count, "test": test_count}


def split_records_by_signer(records: List[DatasetRecord], seed: int = 42) -> Dict[str, List[DatasetRecord]]:
    """Splits accepted samples by signer so clips from one signer stay in one split."""
    unique_signers = sorted({record.signer_id for record in records})
    if not unique_signers:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    rng.shuffle(unique_signers)

    counts = _allocate_signer_counts(len(unique_signers))
    train_signers = set(unique_signers[:counts["train"]])
    val_start = counts["train"]
    val_end = val_start + counts["val"]
    val_signers = set(unique_signers[val_start:val_end])
    test_signers = set(unique_signers[val_end:])

    split_map = {"train": [], "val": [], "test": []}
    for record in records:
        if record.signer_id in train_signers:
            split_map["train"].append(record)
        elif record.signer_id in val_signers:
            split_map["val"].append(record)
        else:
            split_map["test"].append(record)

    return split_map
