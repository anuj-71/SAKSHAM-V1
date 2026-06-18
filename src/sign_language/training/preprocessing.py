from typing import Dict, Optional

import numpy as np

from src.sign_language.training.normalization import normalize_hand_sequence


def build_presence_mask(left_present: np.ndarray, right_present: np.ndarray) -> np.ndarray:
    left = np.asarray(left_present, dtype=np.float32).reshape(-1, 1)
    right = np.asarray(right_present, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([left, right], axis=1)


def stack_hand_sequences(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    left = np.asarray(left_hand, dtype=np.float32)
    right = np.asarray(right_hand, dtype=np.float32)
    return np.stack([left, right], axis=1)


def flatten_frame_features(stacked_hands: np.ndarray, presence_mask: np.ndarray, include_presence: bool = True) -> np.ndarray:
    features = stacked_hands.reshape(stacked_hands.shape[0], -1)
    if include_presence:
        features = np.concatenate([features, presence_mask.astype(np.float32)], axis=1)
    return features.astype(np.float32)


def pad_or_truncate_sequence(features: np.ndarray, target_length: int) -> Dict[str, np.ndarray]:
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    frame_count, feature_dim = features.shape
    if frame_count >= target_length:
        return {
            "features": features[:target_length].astype(np.float32),
            "sequence_mask": np.ones(target_length, dtype=np.float32),
            "original_length": np.int32(frame_count),
        }

    padded = np.zeros((target_length, feature_dim), dtype=np.float32)
    padded[:frame_count] = features
    mask = np.zeros(target_length, dtype=np.float32)
    mask[:frame_count] = 1.0
    return {
        "features": padded,
        "sequence_mask": mask,
        "original_length": np.int32(frame_count),
    }


def preprocess_sample_for_gru(
    sample: Dict[str, np.ndarray],
    target_length: Optional[int] = None,
    include_presence: bool = True,
) -> Dict[str, np.ndarray]:
    """Converts an accepted sample into a normalized frame-major tensor for GRU input."""
    left_hand = np.asarray(sample["left_hand"], dtype=np.float32)
    right_hand = np.asarray(sample["right_hand"], dtype=np.float32)
    left_present = np.asarray(sample["left_present"], dtype=np.float32)
    right_present = np.asarray(sample["right_present"], dtype=np.float32)

    normalized_left = normalize_hand_sequence(left_hand, left_present)
    normalized_right = normalize_hand_sequence(right_hand, right_present)
    stacked = stack_hand_sequences(normalized_left, normalized_right)
    presence_mask = build_presence_mask(left_present, right_present)
    features = flatten_frame_features(stacked, presence_mask, include_presence=include_presence)

    result: Dict[str, np.ndarray] = {
        "features": features,
        "sequence_mask": np.ones(features.shape[0], dtype=np.float32),
        "original_length": np.int32(features.shape[0]),
        "stacked_hands": stacked,
        "presence_mask": presence_mask,
    }
    if target_length is not None:
        padded = pad_or_truncate_sequence(features, target_length)
        result.update(padded)
    return result
