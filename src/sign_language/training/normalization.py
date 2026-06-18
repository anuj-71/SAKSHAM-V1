import numpy as np


WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
EPSILON = 1e-6


def normalize_hand_frame(landmarks: np.ndarray, present: bool) -> np.ndarray:
    """Centers a hand on the wrist and scales by wrist-to-middle-MCP distance."""
    points = np.asarray(landmarks, dtype=np.float32)
    if not present or points.size == 0:
        return np.zeros((21, 3), dtype=np.float32)

    wrist = points[WRIST_INDEX]
    centered = points - wrist
    scale = np.linalg.norm(points[MIDDLE_MCP_INDEX] - wrist)
    if scale < EPSILON:
        scale = 1.0
    return centered / scale


def normalize_hand_sequence(hand_sequence: np.ndarray, presence_mask: np.ndarray) -> np.ndarray:
    """Normalizes a temporal hand sequence frame by frame."""
    frames = np.asarray(hand_sequence, dtype=np.float32)
    mask = np.asarray(presence_mask, dtype=np.float32)
    normalized = np.zeros_like(frames, dtype=np.float32)
    for index in range(frames.shape[0]):
        normalized[index] = normalize_hand_frame(frames[index], bool(mask[index] > 0.5))
    return normalized
