import cv2
import numpy as np
import time
import os
import logging
from typing import Tuple, Optional, List, Dict
from collections import deque
import src.config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
TOOLBAR_X      = 345          # Left edge of toolbar (just right of debug sidebar)
TOOLBAR_W      = 70           # Toolbar width
BTN_H          = 52           # Button height
BTN_GAP        = 6            # Gap between buttons
TOOLBAR_PAD_Y  = 14           # Top padding before first button

HOVER_DWELL_S  = 1.0          # Seconds to hover before selection
SMOOTH_WINDOW  = 12           # Moving-average window (larger = smoother, more lag)
SMOOTH_EMA    = 0.35          # Secondary EMA weight on smoothed point (lower = smoother)
MIN_MOVE_PX    = 6            # Dead-zone: ignore tremors smaller than this (pixels)
INTERP_STEPS   = 6            # Sub-pixel interpolation between consecutive points
PINCH_GRACE_FRAMES = 3        # Frames of non-pinch allowed before breaking stroke

BRUSH_SIZES = {"SMALL": 3, "MEDIUM": 7, "LARGE": 14}
ERASER_SIZES = {"SMALL": 20, "MEDIUM": 40, "LARGE": 65}


class ToolbarButton:
    """A single button in the left-side vertical toolbar."""
    def __init__(self, label: str, y: int, color_bgr: Tuple[int, int, int],
                 action_key: str, display_char: str = ""):
        self.label       = label
        self.y           = y
        self.color_bgr   = color_bgr      # accent / fill colour
        self.action_key  = action_key     # what action this performs
        self.display_char = display_char  # single char shown on button face
        self.hover_time  = 0.0
        self.is_selected = False

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """(x, y, w, h) for hit-testing."""
        return (TOOLBAR_X, self.y, TOOLBAR_W, BTN_H)

    def hit(self, pt: Tuple[int, int]) -> bool:
        x, y, w, h = self.rect
        return x <= pt[0] <= x + w and y <= pt[1] <= y + h

    def update(self, pt: Tuple[int, int], dt: float) -> bool:
        """Returns True the frame the dwell threshold is crossed."""
        if self.hit(pt):
            self.hover_time += dt
            if self.hover_time >= HOVER_DWELL_S:
                self.hover_time = 0.0
                return True          # fired
        else:
            self.hover_time = max(0.0, self.hover_time - dt * 2.0)
        return False

    def progress(self) -> float:
        return min(1.0, self.hover_time / HOVER_DWELL_S)


class Whiteboard:
    """
    Phase-3 redesigned virtual whiteboard.

    Drawing model:
      • POINT  → move cursor (no drawing)
      • PINCH  → draw (or erase) while pinch is held
      • Release → end stroke

    Toolbar:
      Vertical strip on the left.  Hover index-tip for 1 s to select tool.
    """

    # ── Color palette (BGR) ──────────────────────────────────────
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "RED":     (0,   0,   255),
        "GREEN":   (0,   255, 0  ),
        "BLUE":    (255, 0,   0  ),
        "YELLOW":  (0,   255, 255),
        "MAGENTA": (255, 0,   255),
    }

    def __init__(self, width: int = 1280, height: int = 720):
        self.width  = width
        self.height = height

        # Canvas (black = transparent in overlay)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Tool state
        self.color_name  = "RED"
        self.color_bgr   = self.COLORS["RED"]
        self.brush_name  = "MEDIUM"
        self.brush_size  = BRUSH_SIZES["MEDIUM"]
        self.eraser_name = "MEDIUM"
        self.eraser_size = ERASER_SIZES["MEDIUM"]
        self.tool        = "DRAW"        # "DRAW" | "ERASE"

        # Drawing state
        self.is_drawing   = False        # True while pinch held
        self.prev_pt: Optional[Tuple[int, int]] = None
        self.smooth_buf: deque = deque(maxlen=SMOOTH_WINDOW)
        self.ema_pt: Optional[Tuple[int, int]] = None    # secondary EMA tracker
        self.pinch_miss   = 0            # grace-period counter for pinch breaks

        # Eraser cursor position for visual indicator
        self.eraser_pos: Optional[Tuple[int, int]] = None

        # Build toolbar buttons
        self._buttons: List[ToolbarButton] = []
        self._build_toolbar()

        # Save directory
        self._save_dir = os.path.join(
            os.path.expanduser("~"), "Pictures", "AntiGravity"
        )

        logging.info("Whiteboard (Phase 3) initialised.")

    # ── Toolbar construction ─────────────────────────────────────
    def _build_toolbar(self) -> None:
        y = TOOLBAR_PAD_Y

        def add(label, color, key, char=""):
            nonlocal y
            self._buttons.append(ToolbarButton(label, y, color, key, char))
            y += BTN_H + BTN_GAP

        # Colors
        add("RED",     (0,   0,   220), "color_RED",     "R")
        add("GREEN",   (0,   200, 0  ), "color_GREEN",   "G")
        add("BLUE",    (200, 0,   0  ), "color_BLUE",    "B")
        add("YELLOW",  (0,   220, 220), "color_YELLOW",  "Y")
        add("MAGENTA", (200, 0,   200), "color_MAGENTA", "M")

        y += 10   # separator gap

        # Brush sizes
        add("SM",  (160, 160, 160), "brush_SMALL",  "S")
        add("MD",  (200, 200, 200), "brush_MEDIUM", "M")
        add("LG",  (240, 240, 240), "brush_LARGE",  "L")

        y += 10   # separator gap

        # Tools
        add("ERASE", (80, 80, 80),   "tool_ERASE", "E")
        add("CLEAR", (30, 120, 230), "tool_CLEAR", "C")
        add("SAVE",  (0,  180, 80 ), "tool_SAVE",  "S")

        # Select defaults
        self._mark_selected()

    def _mark_selected(self) -> None:
        for b in self._buttons:
            if b.action_key == f"color_{self.color_name}":
                b.is_selected = True
            elif b.action_key == f"brush_{self.brush_name}":
                b.is_selected = True
            elif self.tool == "ERASE" and b.action_key == "tool_ERASE":
                b.is_selected = True

    # ── Public API ───────────────────────────────────────────────
    def set_color(self, name: str) -> None:
        if name in self.COLORS:
            self.color_name = name
            self.color_bgr  = self.COLORS[name]
            self.tool       = "DRAW"
            logging.info(f"Whiteboard color → {name}")

    def set_brush(self, size_name: str) -> None:
        if size_name in BRUSH_SIZES:
            self.brush_name = size_name
            self.brush_size = BRUSH_SIZES[size_name]
            logging.info(f"Whiteboard brush → {size_name} ({self.brush_size}px)")

    def set_tool(self, tool: str) -> None:
        if tool in ("DRAW", "ERASE"):
            self.tool = tool
            logging.info(f"Whiteboard tool → {tool}")

    def clear(self) -> None:
        self.canvas.fill(0)
        self.prev_pt   = None
        self.is_drawing = False
        self.smooth_buf.clear()
        logging.info("Whiteboard canvas cleared.")

    def save(self) -> str:
        os.makedirs(self._save_dir, exist_ok=True)
        ts       = time.strftime("%Y%m%d_%H%M%S")
        filename = f"whiteboard_{ts}.png"
        filepath = os.path.join(self._save_dir, filename)
        cv2.imwrite(filepath, self.canvas)
        logging.info(f"Drawing saved → {filepath}")
        return filepath

    # ── Drawing engine ───────────────────────────────────────────
    def process_drawing(self, index_tip: Tuple[int, int], active_gesture: str) -> None:
        """
        Called every frame.
        - active_gesture == "PINCH"  → draw / erase
        - anything else              → lift pen (with a short grace period)

        Uses a two-stage filter:
          1. Moving average over SMOOTH_WINDOW frames
          2. EMA on the moving-average output
        Plus a minimum-movement dead-zone to kill micro-jitter.
        """
        pinching = (active_gesture == "PINCH")

        # Grace period: ignore up to PINCH_GRACE_FRAMES consecutive non-pinch frames
        # so tiny detection glitches don't break the stroke.
        if pinching:
            self.pinch_miss = 0
        else:
            self.pinch_miss += 1

        if pinching or self.pinch_miss <= PINCH_GRACE_FRAMES:
            # Accumulate into moving-average buffer
            self.smooth_buf.append(index_tip)
            ma = self._smooth_point()          # stage 1: moving average

            # Stage 2: EMA on top of moving average
            if self.ema_pt is None:
                self.ema_pt = ma
            else:
                ex = int(self.ema_pt[0] * (1 - SMOOTH_EMA) + ma[0] * SMOOTH_EMA)
                ey = int(self.ema_pt[1] * (1 - SMOOTH_EMA) + ma[1] * SMOOTH_EMA)
                self.ema_pt = (ex, ey)

            smoothed = self.ema_pt

            if pinching:    # Only paint while genuinely pinching
                if self.prev_pt is None:
                    self.prev_pt    = smoothed
                    self.is_drawing = True
                else:
                    dx   = smoothed[0] - self.prev_pt[0]
                    dy   = smoothed[1] - self.prev_pt[1]
                    dist = (dx * dx + dy * dy) ** 0.5

                    if dist >= MIN_MOVE_PX:
                        pts = self._interpolate(self.prev_pt, smoothed, INTERP_STEPS)
                        for p in pts:
                            self._paint(p)
                        self.prev_pt = smoothed
        else:
            # Pen lifted — clear all state
            self.prev_pt    = None
            self.is_drawing = False
            self.smooth_buf.clear()
            self.ema_pt     = None
            self.pinch_miss = 0

        # Track eraser cursor for visual indicator
        self.eraser_pos = index_tip if (self.tool == "ERASE") else None

    def _smooth_point(self) -> Tuple[int, int]:
        """Returns the moving-average of the smoothing buffer."""
        if not self.smooth_buf:
            return (0, 0)
        xs = [p[0] for p in self.smooth_buf]
        ys = [p[1] for p in self.smooth_buf]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    @staticmethod
    def _interpolate(p1: Tuple[int, int], p2: Tuple[int, int],
                     steps: int) -> List[Tuple[int, int]]:
        """Returns `steps` intermediate points between p1 and p2."""
        pts = []
        for i in range(1, steps + 1):
            t = i / steps
            x = int(p1[0] + (p2[0] - p1[0]) * t)
            y = int(p1[1] + (p2[1] - p1[1]) * t)
            pts.append((x, y))
        return pts

    def _paint(self, pt: Tuple[int, int]) -> None:
        """Applies one brush/eraser stamp at `pt`."""
        if self.tool == "DRAW":
            cv2.circle(self.canvas, pt, self.brush_size, self.color_bgr, -1, cv2.LINE_AA)
        elif self.tool == "ERASE":
            cv2.circle(self.canvas, pt, self.eraser_size, (0, 0, 0), -1)

    # ── Toolbar update ───────────────────────────────────────────
    def update_toolbar(self, index_tip: Tuple[int, int], dt: float) -> Optional[str]:
        """
        Call every frame with the index-finger tip pixel position.
        Returns a notification string if a tool was selected, else None.
        """
        for btn in self._buttons:
            fired = btn.update(index_tip, dt)
            if fired:
                return self._handle_action(btn.action_key)
        return None

    def _handle_action(self, key: str) -> str:
        """Executes a toolbar action and returns a toast message."""
        # Deselect all of same group first
        if key.startswith("color_"):
            name = key.split("_", 1)[1]
            self.set_color(name)
            for b in self._buttons:
                b.is_selected = (b.action_key == key)
            return f"COLOR: {name}"

        if key.startswith("brush_"):
            size = key.split("_", 1)[1]
            self.set_brush(size)
            for b in self._buttons:
                if b.action_key.startswith("brush_"):
                    b.is_selected = (b.action_key == key)
            return f"BRUSH: {size}"

        if key == "tool_ERASE":
            self.set_tool("ERASE")
            for b in self._buttons:
                b.is_selected = (b.action_key == "tool_ERASE")
            return "ERASER ACTIVE"

        if key == "tool_CLEAR":
            self.clear()
            return "CANVAS CLEARED"

        if key == "tool_SAVE":
            path = self.save()
            return f"SAVED: {os.path.basename(path)}"

        return ""

    # ── Rendering ────────────────────────────────────────────────
    def get_overlay(self, frame: cv2.Mat) -> cv2.Mat:
        """Composites canvas drawings onto the webcam frame."""
        gray  = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        out   = frame.copy()
        out[mask > 0] = self.canvas[mask > 0]
        return out

    def draw_toolbar(self, frame: cv2.Mat, index_tip: Optional[Tuple[int, int]] = None) -> None:
        """
        Renders the vertical left-side toolbar onto `frame` in-place.
        `index_tip` is used to draw the eraser circle indicator.
        """
        h = frame.shape[0]

        # Background strip
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (TOOLBAR_X, 0),
                      (TOOLBAR_X + TOOLBAR_W, h),
                      (12, 12, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Vertical dividers
        cv2.line(frame, (TOOLBAR_X, 0), (TOOLBAR_X, h), config.COLOR_BORDER, 1)
        cv2.line(frame, (TOOLBAR_X + TOOLBAR_W, 0),
                 (TOOLBAR_X + TOOLBAR_W, h), config.COLOR_BORDER, 1)

        for btn in self._buttons:
            x, y, w, bh = btn.rect
            prog = btn.progress()

            # Button background
            bg_col = (30, 30, 40)
            if btn.is_selected:
                bg_col = (int(btn.color_bgr[0] * 0.25),
                          int(btn.color_bgr[1] * 0.25),
                          int(btn.color_bgr[2] * 0.25))
            cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + bh - 2), bg_col, -1)

            # Border (bright if selected)
            border_col = btn.color_bgr if btn.is_selected else (60, 60, 80)
            border_thick = 2 if btn.is_selected else 1
            cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + bh - 2),
                          border_col, border_thick, cv2.LINE_AA)

            # Hover progress arc (drawn as a bottom fill bar)
            if prog > 0:
                bar_h = int(bh * prog)
                ov2 = frame.copy()
                cv2.rectangle(ov2, (x + 2, y + bh - bar_h),
                              (x + w - 2, y + bh - 2),
                              btn.color_bgr, -1)
                cv2.addWeighted(ov2, 0.35, frame, 0.65, 0, frame)

            # Label text (centred)
            font_scale = 0.38 if len(btn.label) <= 2 else 0.32
            ts = cv2.getTextSize(btn.label, config.FONT_STYLE, font_scale, 1)[0]
            tx = x + (w - ts[0]) // 2
            ty = y + (bh + ts[1]) // 2
            text_col = btn.color_bgr if not btn.is_selected else (255, 255, 255)
            cv2.putText(frame, btn.label, (tx, ty),
                        config.FONT_STYLE, font_scale, text_col, 1, cv2.LINE_AA)

        # Eraser visual indicator (large circle at fingertip)
        if self.tool == "ERASE" and index_tip is not None:
            cx, cy = index_tip
            cv2.circle(frame, (cx, cy), self.eraser_size,
                       (120, 120, 120), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (200, 200, 200), -1, cv2.LINE_AA)

        # Active tool info strip at the bottom of toolbar
        self._draw_status_strip(frame)

    def _draw_status_strip(self, frame: cv2.Mat) -> None:
        """Small status strip at the bottom of the toolbar column."""
        h = frame.shape[0]
        sx = TOOLBAR_X + 4
        sy = h - 60

        col = self.color_bgr if self.tool == "DRAW" else (120, 120, 120)
        label = self.color_name if self.tool == "DRAW" else "ERASE"
        bs    = self.brush_name[0]  # S / M / L

        cv2.putText(frame, label, (sx, sy),
                    config.FONT_STYLE, 0.32, col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Sz:{bs}", (sx, sy + 14),
                    config.FONT_STYLE, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

        # Color swatch
        if self.tool == "DRAW":
            cv2.rectangle(frame, (sx, sy + 20), (sx + 30, sy + 34),
                          self.color_bgr, -1)
