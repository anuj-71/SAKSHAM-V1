import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import src.config as config

class HoverButton:
    """
    A holographic virtual button that triggers a callback when the user's
    cursor dwells inside its bounding box for a threshold duration (0.5s).
    """
    def __init__(self, label: str, x: int, y: int, w: int, h: int, callback, 
                 mode_restriction: Optional[str] = None, color: Tuple[int, int, int] = None):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.callback = callback
        self.mode_restriction = mode_restriction  # Only active in this mode (None = always)
        self.color = color if color else config.COLOR_BORDER
        self.hover_time = 0.0
        self.is_selected = False

    def check_hover(self, cursor_pos: Tuple[int, int], dt: float) -> bool:
        """Checks if cursor is inside boundaries. Increments hover progress. Triggers callback at threshold."""
        cx, cy = cursor_pos
        in_bounds = (self.x <= cx <= self.x + self.w) and (self.y <= cy <= self.y + self.h)
        
        if in_bounds:
            self.hover_time += dt
            if self.hover_time >= config.HOVER_ACTIVATION_TIME_S:
                self.hover_time = 0.0  # Reset on click
                self.callback()
                return True
        else:
            # Gradual decay of progress makes hover forgiving to coordinate noise
            self.hover_time = max(0.0, self.hover_time - dt * 2.0)
            
        return False

    def get_progress(self) -> float:
        return min(1.0, self.hover_time / config.HOVER_ACTIVATION_TIME_S)

class UIManager:
    """
    Renders the hand skeleton overlay and a high-tech Gesture Debug Dashboard
    directly onto the video frame using OpenCV.
    """
    def __init__(self):
        # MediaPipe connections indices for hand skeleton
        self.mp_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (9, 10), (10, 11), (11, 12),         # Middle (part 1)
            (13, 14), (14, 15), (15, 16),        # Ring (part 1)
            (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (5, 9), (9, 13), (13, 17)            # Palm knuckles base
        ]
        
        # Virtual Hover Buttons
        self.buttons: List[HoverButton] = []
        
        # Toast Notifications
        self.toast_msg = ""
        self.toast_expiry = 0.0

    def draw_hud(self, frame: cv2.Mat, hand_detected: bool, hand_data: Dict,
                 active_gesture: str, gesture_metrics: Dict, fps: float,
                 cursor_pos: Tuple[int, int] = (0, 0), is_clicked: bool = False,
                 mouse_diagnostics: Dict = None, active_mode: str = "MOUSE",
                 whiteboard=None, index_tip: Optional[Tuple[int, int]] = None,
                 thumb_hold_progress: float = 0.0,
                 dt: float = 0.033) -> cv2.Mat:
        """
        Draws the full HUD including Debug Dashboard sidebar, whiteboard toolbar,
        and the hand skeleton overlay.  Returns the annotated frame.
        """
        output_frame = frame.copy()

        if config.DISABLE_HUD_EFFECTS:
            fps_str = f"FPS: {fps:.1f} | GESTURE: {active_gesture} | MODE: {active_mode} (HUD: OFF)"
            cv2.rectangle(output_frame, (10, 10), (420, 40), (0, 0, 0), -1)
            cv2.putText(output_frame, fps_str, (15, 30),
                        config.FONT_STYLE, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # 1. Debug Dashboard Sidebar (left panel)
            output_frame = self._draw_sidebar(
                output_frame, hand_detected, hand_data, active_gesture,
                gesture_metrics, fps, cursor_pos, is_clicked, mouse_diagnostics,
                active_mode=active_mode, whiteboard=whiteboard
            )

            # 2. Whiteboard toolbar (vertical strip, only in WHITEBOARD mode)
            if active_mode == "WHITEBOARD" and whiteboard is not None:
                whiteboard.draw_toolbar(output_frame, index_tip)

        # 3. Hand skeleton overlay
        if hand_detected and hand_data:
            self._draw_hand_skeleton(output_frame, hand_data)

        # 4. THUMBS_UP hold progress bar (shown whenever charging)
        if thumb_hold_progress > 0:
            self._draw_thumbs_hold_bar(output_frame, thumb_hold_progress, active_mode)

        # 5. Toast notifications
        if time.time() < self.toast_expiry and self.toast_msg:
            self._draw_toast(output_frame)

        return output_frame

    def _draw_thumbs_hold_bar(self, frame: cv2.Mat, progress: float, mode: str) -> None:
        """
        Renders a horizontal hold-progress bar near the bottom of the camera feed.
        Fills left→right as the user holds THUMBS_UP.
        """
        h, w, _ = frame.shape
        sidebar_w = 340
        cam_x     = sidebar_w + 40
        bar_y     = h - 38
        bar_w     = w - sidebar_w - 80
        bar_h     = 18

        action = "EXIT WHITEBOARD" if mode == "WHITEBOARD" else "ENTER WHITEBOARD"
        label  = f"\U0001f44d HOLD {int(progress * 100)}%  \u2192  {action}"

        # Dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (cam_x - 5, bar_y - 22),
                      (cam_x + bar_w + 5, bar_y + bar_h + 4), (8, 8, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Track (empty)
        cv2.rectangle(frame, (cam_x, bar_y), (cam_x + bar_w, bar_y + bar_h),
                      (50, 50, 70), -1)
        cv2.rectangle(frame, (cam_x, bar_y), (cam_x + bar_w, bar_y + bar_h),
                      config.COLOR_TEXT_MUTED, 1)

        # Fill (neon cyan)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            fill_col = config.COLOR_BORDER
            cv2.rectangle(frame, (cam_x, bar_y), (cam_x + fill_w, bar_y + bar_h),
                          fill_col, -1)
            # Bright edge glow at fill boundary
            cv2.line(frame, (cam_x + fill_w, bar_y),
                     (cam_x + fill_w, bar_y + bar_h), (255, 255, 255), 1)

        # Label text above the bar
        ts = cv2.getTextSize(label, config.FONT_STYLE, 0.4, 1)[0]
        tx = cam_x + (bar_w - ts[0]) // 2
        cv2.putText(frame, label, (tx, bar_y - 6),
                    config.FONT_STYLE, 0.4, config.COLOR_BORDER, 1, cv2.LINE_AA)

    def _draw_sidebar(self, frame: cv2.Mat, hand_detected: bool, hand_data: Dict,
                      active_gesture: str, gesture_metrics: Dict, fps: float,
                      cursor_pos: Tuple[int, int] = (0, 0), is_clicked: bool = False,
                      mouse_diagnostics: Dict = None,
                      active_mode: str = "MOUSE", whiteboard=None) -> cv2.Mat:
        """Draws the transparent background panel and all debug metrics text."""

        h, w, _ = frame.shape
        sidebar_w = 340
        
        # Create a copy for blending the transparent background
        overlay = frame.copy()
        
        # Draw background panel (very dark grey-blue)
        cv2.rectangle(overlay, (0, 0), (sidebar_w, h), config.COLOR_BG, -1)
        # Draw border line separating sidebar from main camera feed
        cv2.line(overlay, (sidebar_w, 0), (sidebar_w, h), config.COLOR_BORDER, 2)
        
        # Blend the overlay to make the sidebar semi-transparent
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw HUD elements text
        y_pos = 35
        
        # Header Title
        cv2.putText(frame, "ANTIGRAVITY // DEBUG OS", (15, y_pos), 
                    config.FONT_STYLE, 0.6, config.COLOR_BORDER, 2, cv2.LINE_AA)
        y_pos += 20
        cv2.line(frame, (15, y_pos), (sidebar_w - 15, y_pos), config.COLOR_TEXT_MUTED, 1)
        
        # No mode-switch buttons needed — THUMBS_UP gesture is the toggle
        y_pos += 20

        # Gesture shortcut hint strip
        hint_y = 75
        cv2.putText(frame, "\U0001f44d THUMBS UP: Toggle Whiteboard", (15, hint_y),
                    config.FONT_STYLE, 0.34, config.COLOR_BORDER, 1, cv2.LINE_AA)
        cv2.putText(frame, "PINCH: Draw / Click", (15, hint_y + 14),
                    config.FONT_STYLE, 0.34, config.COLOR_SUCCESS, 1, cv2.LINE_AA)
        cv2.line(frame, (15, hint_y + 23), (sidebar_w - 15, hint_y + 23), config.COLOR_TEXT_MUTED, 1)
        y_pos = hint_y + 32
        
        # 1. Performance Section
        cv2.putText(frame, "SYSTEM METRICS", (15, y_pos),
                    config.FONT_STYLE, 0.45, config.COLOR_BORDER, 1, cv2.LINE_AA)
        y_pos += 25

        # Active mode badge
        mode_col = config.COLOR_BORDER if active_mode == "MOUSE" else (0, 220, 120)
        cv2.putText(frame, f"Mode:         {active_mode}", (25, y_pos),
                    config.FONT_STYLE, 0.5, mode_col, 1, cv2.LINE_AA)
        y_pos += 20

        # If in whiteboard, show color + brush
        if active_mode == "WHITEBOARD" and whiteboard is not None:
            col_bgr = whiteboard.color_bgr if whiteboard.tool == "DRAW" else (120, 120, 120)
            tool_label = whiteboard.color_name if whiteboard.tool == "DRAW" else "ERASER"
            cv2.putText(frame, f"WB Tool:      {whiteboard.tool}", (25, y_pos),
                        config.FONT_STYLE, 0.45, col_bgr, 1, cv2.LINE_AA)
            y_pos += 18
            cv2.putText(frame, f"Color:        {tool_label}", (25, y_pos),
                        config.FONT_STYLE, 0.45, col_bgr, 1, cv2.LINE_AA)
            y_pos += 18
            cv2.putText(frame, f"Brush:        {whiteboard.brush_name}", (25, y_pos),
                        config.FONT_STYLE, 0.45, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            y_pos += 18

        fps_color = config.COLOR_SUCCESS if fps >= 30 else config.COLOR_WARNING
        cv2.putText(frame, f"System FPS:   {fps:.1f}", (25, y_pos),
                    config.FONT_STYLE, 0.5, fps_color, 1, cv2.LINE_AA)
        y_pos += 20

        status_text  = "CONNECTED" if hand_detected else "SCANNING..."
        status_color = config.COLOR_SUCCESS if hand_detected else config.COLOR_WARNING
        cv2.putText(frame, f"Webcam:       {status_text}", (25, y_pos),
                    config.FONT_STYLE, 0.5, status_color, 1, cv2.LINE_AA)
        y_pos += 25
        cv2.line(frame, (15, y_pos), (sidebar_w - 15, y_pos), config.COLOR_TEXT_MUTED, 1)
        y_pos += 30

        
        # 2. Gesture Engine Section
        cv2.putText(frame, "GESTURE ENGINE", (15, y_pos), 
                    config.FONT_STYLE, 0.45, config.COLOR_BORDER, 1, cv2.LINE_AA)
        y_pos += 25
        
        # Current Gesture Display with high-contrast box
        cv2.putText(frame, "Active Gesture:", (25, y_pos), 
                    config.FONT_STYLE, 0.5, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        y_pos += 20
        
        # Draw background highlight for the active gesture
        g_color = config.COLOR_SUCCESS if active_gesture != "NONE" else config.COLOR_TEXT_MUTED
        cv2.rectangle(frame, (25, y_pos - 15), (sidebar_w - 25, y_pos + 12), (g_color[0]//5, g_color[1]//5, g_color[2]//5), -1)
        cv2.rectangle(frame, (25, y_pos - 15), (sidebar_w - 25, y_pos + 12), g_color, 1)
        cv2.putText(frame, f" {active_gesture}", (35, y_pos + 4), 
                    config.FONT_STYLE, 0.65, g_color, 2, cv2.LINE_AA)
        y_pos += 35
        
        # Tracking Confidence
        conf = gesture_metrics.get("confidence", 0.0)
        cv2.putText(frame, f"Tracking Conf: {conf*100:.1f}%", (25, y_pos), 
                    config.FONT_STYLE, 0.5, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        y_pos += 20
        
        # Hand Label (Left/Right)
        hand_lbl = hand_data.get("label", "N/A")
        cv2.putText(frame, f"Active Hand:   {hand_lbl}", (25, y_pos), 
                    config.FONT_STYLE, 0.5, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        y_pos += 20
        
        # Pinch Distance Ratio and Meter
        pinch_dist = gesture_metrics.get("pinch_distance", 1.0)
        pinch_thresh = gesture_metrics.get("pinch_threshold", 0.25)
        cv2.putText(frame, f"Pinch Ratio:   {pinch_dist:.2f} / {pinch_thresh:.2f}", (25, y_pos), 
                    config.FONT_STYLE, 0.5, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        y_pos += 10
        
        # Draw visual neon pinch meter bar
        bar_x, bar_y = 25, y_pos
        bar_w, bar_h = sidebar_w - 50, 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), config.COLOR_BG, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), config.COLOR_TEXT_MUTED, 1)
        
        # Fill ratio increases as pinch distance decreases (squeeze to fill)
        fill_ratio = max(0.0, min(1.0, 1.0 - (pinch_dist / 1.0)))
        fill_w = int(bar_w * fill_ratio)
        bar_color = config.COLOR_SUCCESS if pinch_dist < pinch_thresh else config.COLOR_ACCENT
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        
        y_pos += 20
        cv2.line(frame, (15, y_pos), (sidebar_w - 15, y_pos), config.COLOR_TEXT_MUTED, 1)
        y_pos += 30
        
        # 3. Finger States Section
        cv2.putText(frame, "FINGER STATES", (15, y_pos), 
                    config.FONT_STYLE, 0.45, config.COLOR_BORDER, 1, cv2.LINE_AA)
        y_pos += 25
        
        f_states = gesture_metrics.get("finger_states", {})
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        for f in fingers:
            is_up = f_states.get(f, False)
            state_str = "EXTENDED" if is_up else "FOLDED"
            state_color = config.COLOR_SUCCESS if is_up else config.COLOR_ACCENT
            
            cv2.putText(frame, f"{f.capitalize():<10}:", (25, y_pos), 
                        config.FONT_STYLE, 0.5, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, state_str, (140, y_pos), 
                        config.FONT_STYLE, 0.5, state_color, 1, cv2.LINE_AA)
            y_pos += 20
            
        y_pos += 10
        cv2.line(frame, (15, y_pos), (sidebar_w - 15, y_pos), config.COLOR_TEXT_MUTED, 1)
        y_pos += 30
        
        # 4. Virtual Mouse & Coordinates Section
        cv2.putText(frame, "MOUSE & DIAGNOSTICS", (15, y_pos), 
                    config.FONT_STYLE, 0.45, config.COLOR_BORDER, 1, cv2.LINE_AA)
        y_pos += 25
        
        if mouse_diagnostics:
            screen_res = mouse_diagnostics.get("screen_res", "N/A")
            alpha = mouse_diagnostics.get("smoothing_alpha", config.MOUSE_EMA_ALPHA)
            m_down = mouse_diagnostics.get("mouse_down", False)
            drag = mouse_diagnostics.get("drag_active", False)
            dbl_timer = mouse_diagnostics.get("double_click_window", 0.0)
            last_trans = mouse_diagnostics.get("last_transition", "NONE")
            win32_status = mouse_diagnostics.get("win32_status", "IDLE")
            
            # Cursor pos
            cv2.putText(frame, "Cursor Screen:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"X: {cursor_pos[0]:<4} Y: {cursor_pos[1]:<4}", (150, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            y_pos += 18
            
            # Mapped resolution
            cv2.putText(frame, "Display Res:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, screen_res, (150, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            y_pos += 18
            
            # Smoothing Alpha
            cv2.putText(frame, "Smooth Alpha:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{alpha:.3f}", (150, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            y_pos += 18
            
            # Click state
            click_str = "DOWN" if m_down else "UP"
            click_color = config.COLOR_SUCCESS if m_down else config.COLOR_TEXT_MUTED
            cv2.putText(frame, "Mouse Key:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, click_str, (150, y_pos), 
                        config.FONT_STYLE, 0.45, click_color, 1, cv2.LINE_AA)
            y_pos += 18
            
            # Drag state
            drag_str = "ACTIVE" if drag else "INACTIVE"
            drag_color = config.COLOR_SUCCESS if drag else config.COLOR_TEXT_MUTED
            cv2.putText(frame, "Drag State:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, drag_str, (150, y_pos), 
                        config.FONT_STYLE, 0.45, drag_color, 1, cv2.LINE_AA)
            y_pos += 18
            
            # Double Click Window
            dbl_str = f"ACTIVE ({dbl_timer:.2f}s)" if dbl_timer > 0 else "INACTIVE"
            dbl_color = config.COLOR_WARNING if dbl_timer > 0 else config.COLOR_TEXT_MUTED
            cv2.putText(frame, "DblClick Win:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, dbl_str, (150, y_pos), 
                        config.FONT_STYLE, 0.45, dbl_color, 1, cv2.LINE_AA)
            y_pos += 18
            
            # Last Transition & Win32 Status
            cv2.putText(frame, "Transition:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, last_trans, (150, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_WARNING, 1, cv2.LINE_AA)
            y_pos += 18
            
            cv2.putText(frame, "Win32 Call:", (25, y_pos), 
                        config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, win32_status, (150, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_SUCCESS if "SUCCESS" in win32_status else config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            y_pos += 20
        else:
            cv2.putText(frame, "MOUSE INACTIVE", (25, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_WARNING, 1, cv2.LINE_AA)
            y_pos += 20
            
        # Landmark coordinate details if hand is present
        if hand_detected and "pixel_landmarks" in hand_data:
            pixel_lms = hand_data["pixel_landmarks"]
            
            coords = [
                ("WRIST [0]", pixel_lms[0]),
                ("INDEX TIP [8]", pixel_lms[8])
            ]
            
            for label, pt in coords:
                cv2.putText(frame, f"{label}:", (25, y_pos), 
                            config.FONT_STYLE, 0.42, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(frame, f"X: {pt[0]:<4} Y: {pt[1]:<4}", (150, y_pos), 
                            config.FONT_STYLE, 0.45, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
                y_pos += 18
        else:
            cv2.putText(frame, "NO TRACKING DATA", (25, y_pos), 
                        config.FONT_STYLE, 0.45, config.COLOR_WARNING, 1, cv2.LINE_AA)
            y_pos += 18
            
        return frame

    def _draw_hand_skeleton(self, frame: cv2.Mat, hand_data: Dict) -> None:
        """Draws custom neon lines and joints overlay on the detected hand."""
        pixel_pts = hand_data["pixel_landmarks"]
        
        # 1. Draw connection lines
        for start_idx, end_idx in self.mp_connections:
            if start_idx < len(pixel_pts) and end_idx < len(pixel_pts):
                p1 = pixel_pts[start_idx]
                p2 = pixel_pts[end_idx]
                cv2.line(frame, p1, p2, config.COLOR_SKELETON_LINES, 2, cv2.LINE_AA)
                
        # 2. Draw joints (landmarks)
        for i, pt in enumerate(pixel_pts):
            # Define specific colors for index finger tip (cursor controller)
            if i == 8:
                # Neon Cyan outer ring, white center
                cv2.circle(frame, pt, 7, config.COLOR_SUCCESS, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
            elif i == 4:
                # Neon Orange for thumb tip (click indicator)
                cv2.circle(frame, pt, 7, config.COLOR_WARNING, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # standard neon joints
                cv2.circle(frame, pt, 5, config.COLOR_SKELETON_JOINTS, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
                
        # 3. Draw a glowing bounding box around the hand
        if not config.DISABLE_HUD_EFFECTS:
            x_min, y_min, x_max, y_max = self.get_bbox_coords(pixel_pts, frame.shape)
            # Draw decorative cyberpunk corner brackets for the hand boundary box
            self._draw_corners(frame, (x_min, y_min, x_max - x_min, y_max - y_min), 12, config.COLOR_BORDER, 1)

    def get_bbox_coords(self, pixel_landmarks: List[Tuple[int, int]], shape: Tuple[int, int, int], padding: int = 15) -> Tuple[int, int, int, int]:
        """Calculates hand bounding box dimensions bounded by frame size."""
        h, w, _ = shape
        xs = [pt[0] for pt in pixel_landmarks]
        ys = [pt[1] for pt in pixel_landmarks]
        
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(w, max(xs) + padding)
        y_max = min(h, max(ys) + padding)
        
        return x_min, y_min, x_max, y_max

    def _draw_corners(self, img: cv2.Mat, bbox: Tuple[int, int, int, int], length: int, color: Tuple[int, int, int], thickness: int) -> None:
        """Draws cyberpunk corner brackets around a bounding box."""
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Top Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        
        # Top Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        
        # Bottom Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        
        # Bottom Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def register_buttons(self, buttons: List[HoverButton]) -> None:
        """Saves active holographic buttons in the UI manager."""
        self.buttons = buttons

    def trigger_toast(self, message: str, duration: float = 3.0) -> None:
        """Sets a flashing system message on the user's HUD."""
        self.toast_msg = message
        self.toast_expiry = time.time() + duration

    def _update_and_draw_buttons(self, frame: cv2.Mat, cursor_pos: Tuple[int, int], dt: float, active_mode: str) -> None:
        """Processes hover timers and draws visual progress meters on each HUD button."""
        sidebar_w = 340
        for btn in self.buttons:
            # Check button mode restriction (e.g. whiteboard buttons only shown in Whiteboard mode)
            if btn.mode_restriction is not None and btn.mode_restriction != active_mode:
                continue
                
            # Process hover progress
            btn.check_hover(cursor_pos, dt)
            
            # Select color based on status
            btn_color = btn.color
            if btn.is_selected:
                bg_color = (btn_color[0]//3, btn_color[1]//3, btn_color[2]//3)
                thickness = 2
            else:
                bg_color = config.COLOR_BG
                thickness = 1
                
            # Draw button box
            cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), bg_color, -1)
            cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), btn_color, thickness)
            
            # Draw label centered
            text_size = cv2.getTextSize(btn.label, config.FONT_STYLE, 0.38, 1)[0]
            tx = btn.x + (btn.w - text_size[0]) // 2
            ty = btn.y + (btn.h + text_size[1]) // 2
            cv2.putText(frame, btn.label, (tx, ty), 
                        config.FONT_STYLE, 0.38, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            
            # Render visual fill line showing dwell progress (0.0 to 1.0)
            prog = btn.get_progress()
            if prog > 0:
                bar_w = int(btn.w * prog)
                cv2.line(frame, 
                         (btn.x, btn.y + btn.h - 3), 
                         (btn.x + bar_w, btn.y + btn.h - 3), 
                         config.COLOR_SUCCESS, 3, cv2.LINE_AA)

    def _draw_toast(self, frame: cv2.Mat) -> None:
        """Renders a sleek cyberpunk notification banner at the bottom center."""
        h, w, _ = frame.shape
        center_x = 340 + (w - 340) // 2
        toast_y = h - 55
        toast_w, toast_h = 560, 32
        
        tx1, ty1 = center_x - toast_w // 2, toast_y - toast_h // 2
        tx2, ty2 = center_x + toast_w // 2, toast_y + toast_h // 2
        
        # Draw translucent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (10, 5, 0), -1)
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), config.COLOR_SUCCESS, 1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Render text
        text = f"// SYSTEM NOTIFICATION // {self.toast_msg}"
        text_size = cv2.getTextSize(text, config.FONT_STYLE, 0.38, 1)[0]
        cx = center_x - text_size[0] // 2
        cy = toast_y + text_size[1] // 2
        cv2.putText(frame, text, (cx, cy), 
                    config.FONT_STYLE, 0.38, config.COLOR_SUCCESS, 1, cv2.LINE_AA)

    def draw_gesture_hold_arc(self, frame: cv2.Mat, gesture: str, progress: float, current_mode: str) -> None:
        """
        Renders a large neon charging arc in the bottom-right corner showing
        how close the user is to triggering a mode switch via gesture hold.
        Supports THUMBS_UP (enter WHITEBOARD) and THUMBS_DOWN (exit to MOUSE).
        """
        h, w, _ = frame.shape

        # Pick accent color and label based on gesture
        if gesture == "THUMBS_UP":
            arc_color  = config.COLOR_BORDER      # Neon cyan
            icon_label = "\U0001f44d HOLD: ENTER WHITEBOARD"
        elif gesture == "THUMBS_DOWN":
            arc_color  = config.COLOR_WARNING     # Neon orange
            icon_label = "\U0001f44e HOLD: EXIT TO NAVIGATION"
        else:
            arc_color  = config.COLOR_SUCCESS
            icon_label = "HOLD: MODE SWITCH"

        # Arc geometry (bottom-right corner)
        cx, cy, radius = w - 90, h - 90, 55
        angle_start = -90  # top of circle
        angle_end   = int(angle_start + 360 * progress)

        # Dark translucent background disc
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius + 10, (10, 5, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Track ring (dim)
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 360,
                    config.COLOR_TEXT_MUTED, 2, cv2.LINE_AA)

        # Progress arc (bright neon)
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                    angle_start, angle_end, arc_color, 5, cv2.LINE_AA)

        # Percentage text in centre
        pct = f"{int(progress * 100)}%"
        ts  = cv2.getTextSize(pct, config.FONT_STYLE, 0.55, 2)[0]
        cv2.putText(frame, pct,
                    (cx - ts[0] // 2, cy + ts[1] // 2),
                    config.FONT_STYLE, 0.55, arc_color, 2, cv2.LINE_AA)

        # Label below arc
        ts2 = cv2.getTextSize(icon_label, config.FONT_STYLE, 0.32, 1)[0]
        cv2.putText(frame, icon_label,
                    (cx - ts2[0] // 2, cy + radius + 20),
                    config.FONT_STYLE, 0.32, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

        # Pulsing outer ring while charging
        pulse_r = radius + 6 + int(4 * abs(np.sin(progress * np.pi * 6)))
        cv2.ellipse(frame, (cx, cy), (pulse_r, pulse_r), 0,
                    angle_start, angle_end, arc_color, 1, cv2.LINE_AA)
