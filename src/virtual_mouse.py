import ctypes
import time
import logging
from typing import Dict, Tuple, Optional
import src.config as config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Windows User32 constants for mouse events
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

class VirtualMouse:
    """
    Controls the system cursor using native Win32 API calls via ctypes.
    Maps hand landmarks to screen space with adaptive smoothing and edge stabilization.
    """
    def __init__(self):
        # Fetch physical screen dimensions using Win32 API
        self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        logging.info(f"Virtual Mouse initialized. Screen Resolution: {self.screen_width}x{self.screen_height}")
        
        # Cursor tracking state
        self.prev_x = self.screen_width // 2
        self.prev_y = self.screen_height // 2
        self.first_frame = True
        
        # Click and Drag State Machine
        self.is_left_pressed = False
        self.last_pinch_release_time = 0.0
        self.prev_gesture = "NONE"
        
        # Diagnostics
        self.current_alpha = config.MOUSE_EMA_ALPHA
        self.last_transition = "NONE"
        self.last_action_status = "IDLE"
        
    def update(self, hand_data: Dict, active_gesture: str) -> Tuple[int, int, bool]:
        """
        Processes hand data and active gesture to update cursor location and mouse click states.
        Returns the tuple (cursor_x, cursor_y, is_clicked).
        """
        # Read parameters dynamically from config
        border_x = config.MOUSE_ZONE_BORDER_X
        border_y = config.MOUSE_ZONE_BORDER_Y
        double_click_threshold = config.DOUBLE_CLICK_THRESHOLD_S
        
        if not hand_data or "landmarks" not in hand_data:
            # Safety release: if hand disappears, release any active mouse clicks
            self._release_clicks()
            self.prev_gesture = "NONE"
            self.last_transition = "HAND_LOST"
            return self.prev_x, self.prev_y, False

        # Extract normalized coordinates of the index finger tip (landmark 8)
        index_lm = hand_data["landmarks"][8]
        idx_x, idx_y = index_lm[0], index_lm[1]

        # 1. Edge Stabilization (Active Zone Mapping)
        # Map [border, 1 - border] to [0.0, 1.0]
        mapped_x = (idx_x - border_x) / (1.0 - 2 * border_x)
        mapped_y = (idx_y - border_y) / (1.0 - 2 * border_y)
        
        # Clamp coordinates to screen boundaries
        mapped_x = max(0.0, min(1.0, mapped_x))
        mapped_y = max(0.0, min(1.0, mapped_y))
        
        # Calculate target screen coordinates
        target_x = int(mapped_x * self.screen_width)
        target_y = int(mapped_y * self.screen_height)
        
        # Apply Cursor Speed Multiplier to delta movements to tune sensitivity
        delta_x = (target_x - self.prev_x) * config.MOUSE_SPEED_MULTIPLIER
        delta_y = (target_y - self.prev_y) * config.MOUSE_SPEED_MULTIPLIER
        
        target_x_scaled = int(self.prev_x + delta_x)
        target_y_scaled = int(self.prev_y + delta_y)

        # 2. Adaptive Smoothing (Exponential Moving Average)
        if self.first_frame:
            curr_x, curr_y = target_x, target_y
            self.first_frame = False
            self.current_alpha = config.MOUSE_EMA_ALPHA
        else:
            # Calculate distance moved in screen pixels
            dist = ((target_x_scaled - self.prev_x)**2 + (target_y_scaled - self.prev_y)**2)**0.5
            
            if config.MOUSE_ADAPTIVE_SMOOTHING:
                # If movement is tiny (jitter), use low alpha (highly smoothed).
                # If movement is large, scale up alpha for responsiveness.
                base_alpha = config.MOUSE_EMA_ALPHA
                # Non-linear scaling: alpha increases with movement size
                alpha = base_alpha + min(0.48, (dist / 140.0)**1.3)
            else:
                alpha = config.MOUSE_EMA_ALPHA
                
            self.current_alpha = alpha
            
            curr_x = int(alpha * target_x_scaled + (1 - alpha) * self.prev_x)
            curr_y = int(alpha * target_y_scaled + (1 - alpha) * self.prev_y)

        # Clamp final coordinates to screen resolution
        curr_x = max(0, min(self.screen_width - 1, curr_x))
        curr_y = max(0, min(self.screen_height - 1, curr_y))

        # 3. Apply Cursor Movement
        if active_gesture in ["POINT", "PINCH"]:
            ctypes.windll.user32.SetCursorPos(curr_x, curr_y)
            self.prev_x, self.prev_y = curr_x, curr_y

        # 4. Air Click System (State Transitions)
        is_clicked = self.is_left_pressed
        now = time.time()
        
        # Detect Pinch Down (transition to PINCH)
        if active_gesture == "PINCH" and self.prev_gesture != "PINCH":
            time_since_release = now - self.last_pinch_release_time
            if time_since_release < double_click_threshold:
                self._double_click(curr_x, curr_y)
                self.last_transition = "DOUBLE_PINCH"
            else:
                self._press_left_click(curr_x, curr_y)
                self.last_transition = "PINCH_START"
            is_clicked = True
            
        # Detect Pinch Up (transition away from PINCH)
        elif active_gesture != "PINCH" and self.prev_gesture == "PINCH":
            self._release_left_click(curr_x, curr_y)
            self.last_pinch_release_time = now
            self.last_transition = "PINCH_RELEASE"
            is_clicked = False
            
        # Maintain drag if already pressed
        elif active_gesture == "PINCH" and self.prev_gesture == "PINCH":
            self.last_transition = "PINCH_HOLD"
            is_clicked = True
        else:
            self.last_transition = "NONE"

        self.prev_gesture = active_gesture
        return self.prev_x, self.prev_y, is_clicked

    def _press_left_click(self, x: int, y: int) -> None:
        """Simulates left mouse button press (Start click/drag)."""
        if not self.is_left_pressed:
            logging.info(f"[Win32 Action] MOUSEEVENTF_LEFTDOWN at ({x}, {y})")
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.is_left_pressed = True
            self.last_action_status = "LEFTDOWN_SUCCESS"

    def _release_left_click(self, x: int, y: int) -> None:
        """Simulates left mouse button release (End click/drag)."""
        if self.is_left_pressed:
            logging.info(f"[Win32 Action] MOUSEEVENTF_LEFTUP at ({x}, {y})")
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.is_left_pressed = False
            self.last_action_status = "LEFTUP_SUCCESS"

    def _double_click(self, x: int, y: int) -> None:
        """Simulates a left mouse double click."""
        logging.info(f"[Win32 Action] DOUBLE_CLICK at ({x}, {y})")
        # First click
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        # Second click
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.is_left_pressed = False
        self.last_action_status = "DOUBLECLICK_SUCCESS"

    def _release_clicks(self) -> None:
        """Helper to release clicks if tracking is lost to prevent locking mouse."""
        if self.is_left_pressed:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.is_left_pressed = False
            self.last_action_status = "SAFETY_RELEASE_SUCCESS"
            logging.info("[Win32 Action] Safety release: MOUSEEVENTF_LEFTUP fired.")

    def get_diagnostics(self) -> Dict:
        """Returns detailed virtual mouse states for rendering on HUD."""
        now = time.time()
        time_since_release = now - self.last_pinch_release_time
        double_click_window_active = time_since_release < config.DOUBLE_CLICK_THRESHOLD_S
        
        return {
            "screen_res": f"{self.screen_width}x{self.screen_height}",
            "smoothing_alpha": self.current_alpha,
            "mouse_down": self.is_left_pressed,
            "drag_active": self.is_left_pressed and (self.last_transition == "PINCH_HOLD"),
            "double_click_window": max(0.0, config.DOUBLE_CLICK_THRESHOLD_S - time_since_release) if double_click_window_active else 0.0,
            "double_click_ready": double_click_window_active,
            "last_transition": self.last_transition,
            "win32_status": self.last_action_status
        }
