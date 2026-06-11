import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import src.config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandTracker:
    """
    Wraps Google MediaPipe Hands tracking. Detects hand landmarks,
    extracts coordinates, and determines finger extension states.
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Hands model with specified confidence thresholds
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # MediaPipe Landmark connection mapping
        self.connections = self.mp_hands.HAND_CONNECTIONS
        logging.info("MediaPipe Hands initialized with confidence thresholds >= 0.8.")

    def process_frame(self, frame: cv2.Mat) -> Tuple[bool, Dict]:
        """
        Processes a single BGR image frame.
        Returns a tuple of (hand_detected, hand_data_dictionary).
        """
        # Downsample the frame for MediaPipe tracking if specified
        h, w, _ = frame.shape
        
        if config.TRACKING_FRAME_WIDTH > 0 and config.TRACKING_FRAME_HEIGHT > 0:
            small_bgr = cv2.resize(frame, (config.TRACKING_FRAME_WIDTH, config.TRACKING_FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        results = self.hands.process(rgb_frame)
        
        hand_data = {}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # We only track the primary hand (MAX_NUM_HANDS = 1)
            landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0]
            
            label = handedness.classification[0].label  # "Left" or "Right"
            confidence = handedness.classification[0].score
            
            # Extract normalized (0.0 to 1.0) and pixel (scaled to original dimensions) coordinates
            raw_landmarks = []
            pixel_landmarks = []
            for lm in landmarks.landmark:
                raw_landmarks.append((lm.x, lm.y, lm.z))
                pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))
            
            # Determine which fingers are extended (Up = True, Down = False)
            finger_states = self._get_finger_states(raw_landmarks, label)
            
            hand_data = {
                "label": label,                     # "Left" or "Right"
                "confidence": confidence,           # Tracking confidence
                "landmarks": raw_landmarks,         # Normalized coordinates
                "pixel_landmarks": pixel_landmarks, # Pixel coordinates
                "finger_states": finger_states,     # Dict of finger -> True/False
                "raw_mp_landmarks": landmarks       # Raw mediapipe object for drawing
            }
            return True, hand_data
            
        return False, {}

    def _get_finger_states(self, landmarks: List[Tuple[float, float, float]], handedness: str) -> Dict[str, bool]:
        """
        Analyzes landmarks to determine if individual fingers are extended (True) or folded (False).
        Returns a dictionary mapping: 'thumb', 'index', 'middle', 'ring', 'pinky' -> bool
        """
        states = {}
        
        # Helper function to calculate Euclidean distance in 2D space
        def dist_2d(p1_idx: int, p2_idx: int) -> float:
            x1, y1 = landmarks[p1_idx][0], landmarks[p1_idx][1]
            x2, y2 = landmarks[p2_idx][0], landmarks[p2_idx][1]
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
        # 1. Thumb State
        # Robust rotation-independent heuristic: compare distance between thumb tip (4) and pinky base (17)
        # to distance between thumb base MCP (2) and pinky base (17).
        # Also compare thumb tip to index knuckle (5).
        # If thumb tip is extended, it should be further from the pinky MCP than the thumb knuckle.
        d_tip_pinky = dist_2d(4, 17)
        d_mcp_pinky = dist_2d(2, 17)
        
        # In addition, check if the thumb is extended horizontally away from the hand
        # If palm is facing camera: Right hand thumb is on the left (x4 < x2), Left hand thumb is on the right (x4 > x2)
        # We can also check if knuckle 5 is to the right of knuckle 17 to determine palm orientation.
        knuckles_right = landmarks[5][0] > landmarks[17][0]
        
        if knuckles_right: # Palm facing camera for Right hand, or back of hand for Left hand
            is_extended_horiz = landmarks[4][0] < landmarks[3][0] if handedness == "Right" else landmarks[4][0] > landmarks[3][0]
        else: # Back of hand for Right hand, or palm facing camera for Left hand
            is_extended_horiz = landmarks[4][0] > landmarks[3][0] if handedness == "Right" else landmarks[4][0] < landmarks[3][0]
            
        # Thumb is extended if the distance check passes OR the horizontal extension check passes
        states["thumb"] = (d_tip_pinky > d_mcp_pinky * 1.1) or is_extended_horiz
        
        # 2. Four Fingers (Index, Middle, Ring, Pinky)
        # Extended if the tip is above the PIP joint (y-coordinate is smaller in image space)
        # We also check the DIP joint to ensure the finger is straight.
        # Index: Tip (8), PIP (6), MCP (5)
        states["index"] = landmarks[8][1] < landmarks[6][1]
        
        # Middle: Tip (12), PIP (10), MCP (9)
        states["middle"] = landmarks[12][1] < landmarks[10][1]
        
        # Ring: Tip (16), PIP (14), MCP (13)
        states["ring"] = landmarks[16][1] < landmarks[14][1]
        
        # Pinky: Tip (20), PIP (18), MCP (17)
        states["pinky"] = landmarks[20][1] < landmarks[18][1]
        
        return states

    def get_hand_bbox(self, pixel_landmarks: List[Tuple[int, int]], padding: int = 15) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (x_min, y_min, x_max, y_max) around the hand landmarks."""
        xs = [pt[0] for pt in pixel_landmarks]
        ys = [pt[1] for pt in pixel_landmarks]
        
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = max(x_min, max(xs) + padding)
        y_max = max(y_min, max(ys) + padding)
        
        return x_min, y_min, x_max, y_max
