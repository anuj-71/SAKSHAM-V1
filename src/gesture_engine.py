import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
import src.config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GestureEngine:
    """
    Analyzes hand tracking data to detect static and dynamic hand gestures.
    Implements a strict priority hierarchy:
    PINCH > THUMBS_UP > THUMBS_DOWN > PEACE_SIGN > POINT > OPEN_PALM > CLOSED_FIST
    """
    def __init__(self):
        # Buffer for tracking palm centers for swipe detection
        # Elements are tuples: (timestamp, (x_px, y_px))
        self.motion_history: List[Tuple[float, Tuple[int, int]]] = []
        
        # Cooldown management for swipes
        self.last_swipe_time = 0.0
        self.last_swipe_direction: Optional[str] = None
        self.swipe_display_end_time = 0.0  # Show swipe text on HUD for a short duration
        
        # Temporal smoothing for static gestures to prevent flickering
        self.gesture_buffer: List[str] = []
        self.buffer_size = 7  # Number of frames to smooth over (increased for THUMBS_DOWN stability)

    def process_gestures(self, hand_data: Dict) -> Tuple[str, Dict]:
        """
        Analyzes the hand coordinates and finger states.
        Returns a tuple (resolved_gesture, debug_metrics).
        """
        if not hand_data:
            self.motion_history.clear()
            return "NONE", {"confidence": 0.0, "finger_states": {}}

        landmarks = hand_data["landmarks"]
        pixel_landmarks = hand_data["pixel_landmarks"]
        finger_states = hand_data["finger_states"]
        handedness = hand_data["label"]
        
        # Calculate palm center (average of Wrist, Index MCP, Middle MCP, Pinky MCP)
        palm_points = [pixel_landmarks[0], pixel_landmarks[5], pixel_landmarks[9], pixel_landmarks[17]]
        palm_x = int(sum(pt[0] for pt in palm_points) / 4)
        palm_y = int(sum(pt[1] for pt in palm_points) / 4)
        palm_center = (palm_x, palm_y)
        
        # 1. Update Motion History for Swipes
        now = time.time()
        self.motion_history.append((now, palm_center))
        # Keep only history within the time window
        self.motion_history = [
            item for item in self.motion_history 
            if now - item[0] <= config.SWIPE_TIME_LIMIT_S
        ]
        
        # 2. Check for Dynamic Gestures (Swipes)
        detected_swipe = self._detect_swipe(now)
        if detected_swipe:
            self.last_swipe_direction = detected_swipe
            self.last_swipe_time = now
            self.swipe_display_end_time = now + 0.6  # Display swipe on screen for 600ms
            logging.info(f"Swipe detected: {detected_swipe}")
            
        # If we are currently displaying a swipe, we can overlay it,
        # but let's check what the active static gesture is as well.
        
        # 3. Detect Static Gestures
        # Calculate pinch ratio in 2D pixel coordinates:
        # Distance between thumb tip (4) and index tip (8)
        p4 = np.array(pixel_landmarks[4])
        p8 = np.array(pixel_landmarks[8])
        dist_px = float(np.linalg.norm(p4 - p8))
        
        # Reference palm width: distance between index MCP (5) and pinky MCP (17)
        p5 = np.array(pixel_landmarks[5])
        p17 = np.array(pixel_landmarks[17])
        palm_width = float(np.linalg.norm(p5 - p17))
        if palm_width == 0:
            palm_width = 1.0
            
        pinch_ratio = dist_px / palm_width
        
        static_gesture = self._resolve_static_gesture(landmarks, finger_states, pinch_ratio)
        
        # 4. Temporal Smoothing
        self.gesture_buffer.append(static_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
            
        # Find the most frequent gesture in our buffer
        smoothed_gesture = max(set(self.gesture_buffer), key=self.gesture_buffer.count)
        
        # If a swipe was recently detected, let's report the swipe on the dashboard
        active_gesture = smoothed_gesture
        if now < self.swipe_display_end_time and self.last_swipe_direction:
            active_gesture = self.last_swipe_direction

        # Prepare debug metrics
        debug_metrics = {
            "confidence": hand_data["confidence"],
            "finger_states": finger_states,
            "palm_center": palm_center,
            "pinch_distance": pinch_ratio,
            "pinch_threshold": config.PINCH_THRESHOLD,
            "static_gesture": static_gesture,
            "smoothed_gesture": smoothed_gesture,
            "active_gesture": active_gesture,
            "landmark_count": len(landmarks)
        }
        
        return active_gesture, debug_metrics

    def _resolve_static_gesture(self, landmarks: List[Tuple[float, float, float]], finger_states: Dict[str, bool], pinch_ratio: float) -> str:
        """
        Applies logic matching the priority hierarchy to return a single gesture.
        PINCH > THUMBS_UP > THUMBS_DOWN > PEACE_SIGN > POINT > OPEN_PALM > CLOSED_FIST
        """
        # 1. PINCH: Tip of thumb and tip of index are close in 2D pixel ratio
        if pinch_ratio < config.PINCH_THRESHOLD:
            return "PINCH"

        # 2. THUMBS_UP: Thumb extended upward, all other fingers folded
        if (finger_states["thumb"] and
            not finger_states["index"] and
            not finger_states["middle"] and
            not finger_states["ring"] and
            not finger_states["pinky"]):
            # Thumb tip (4) is ABOVE thumb base (2) in image coords (y increases downward)
            if landmarks[4][1] < landmarks[2][1]:
                return "THUMBS_UP"
            # 2b. THUMBS_DOWN: Thumb extended downward, all other fingers folded
            # Thumb tip (4) is BELOW thumb base (2) in image coords
            if landmarks[4][1] > landmarks[2][1]:
                return "THUMBS_DOWN"

        # 3. PEACE_SIGN: Index and Middle extended, Ring and Pinky folded, Thumb folded/neutral
        if (finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            return "PEACE_SIGN"

        # 4. POINT: Index extended, Middle, Ring, Pinky folded
        if (finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            return "POINT"

        # 5. OPEN_PALM: All fingers extended
        if (finger_states["thumb"] and 
            finger_states["index"] and 
            finger_states["middle"] and 
            finger_states["ring"] and 
            finger_states["pinky"]):
            return "OPEN_PALM"

        # 6. CLOSED_FIST: All fingers folded
        if (not finger_states["thumb"] and 
            not finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            return "CLOSED_FIST"

        return "NONE"

    def _detect_swipe(self, current_time: float) -> Optional[str]:
        """
        Analyzes the motion history to detect left or right swipes.
        Returns "SWIPE_LEFT", "SWIPE_RIGHT", or None.
        """
        # Apply cooldown check
        if current_time - self.last_swipe_time < config.SWIPE_COOLDOWN_S:
            return None
            
        if len(self.motion_history) < 5:
            return None
            
        # Get start and end points of the window
        start_time, start_pt = self.motion_history[0]
        end_time, end_pt = self.motion_history[-1]
        
        # Calculate time span
        dt = end_time - start_time
        if dt < 0.1:  # Require at least 100ms of history to judge
            return None
            
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        
        # A swipe is primarily horizontal and fast.
        # Check horizontal distance against threshold.
        # Also ensure vertical movement is less than horizontal movement (horizontal swipe check)
        if abs(dx) >= config.SWIPE_THRESHOLD_PX and abs(dy) < abs(dx) * 0.7:
            # Check monotonic horizontal trend to filter out jitter
            xs = [item[1][0] for item in self.motion_history]
            
            # Simple trend check: if moving right, most points should increase
            # if moving left, most points should decrease
            increases = sum(1 for i in range(len(xs)-1) if xs[i+1] > xs[i])
            ratio = increases / (len(xs) - 1)
            
            if dx > 0 and ratio > 0.6:
                return "SWIPE_RIGHT"
            elif dx < 0 and ratio < 0.4:
                return "SWIPE_LEFT"
                
        return None
