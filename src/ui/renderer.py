import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import src.config.settings as config
from src.session_manager import ConversationSession
from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

class UIRenderer:
    """
    SAKSHAM V1.5 User Interface.
    Utilizes Pillow (PIL) for high-quality text rendering and chat bubbles.

    Layout (top to bottom):
        Header          (60px)
        Conversation    (fills remaining - bottom_section)
        ────────────────────────────────────────────
        Bottom section  (200px):
            Left:  Live Transcript panel  (65% width)
            Right: Camera preview (small) + Session Info
        ────────────────────────────────────────────
        Status bar      (36px)
    """
    def __init__(self, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.width = width
        self.height = height

        # ── Layout constants ──────────────────────────────────────────
        self.header_h = 60
        self.status_h = 36
        self.bottom_h = 200

        self.conv_y = self.header_h
        self.conv_h = self.height - self.header_h - self.bottom_h - self.status_h

        self.bottom_y = self.conv_y + self.conv_h
        self.status_y = self.height - self.status_h

        # Bottom section split: 65% live transcript, 35% camera + stats
        self.live_panel_w = int(self.width * 0.65)
        self.right_panel_x = self.live_panel_w

        # Camera preview – small, just enough to confirm it's working
        self.cam_h = 120
        self.cam_w = int(self.cam_h * 16 / 9)  # ~213px

        # Chat bubble max width: 65% of conversation area width
        self.max_bubble_w_ratio = 0.65

        # ── Fonts (Windows Defaults) ──────────────────────────────────
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"

        if not os.path.exists(font_path):
            font_path = "arial.ttf"
            font_bold_path = "arialbd.ttf"

        try:
            self.font_header = ImageFont.truetype(font_bold_path, 24)
            self.font_subheader = ImageFont.truetype(font_path, 14)
            self.font_body = ImageFont.truetype(font_path, 18)
            self.font_body_bold = ImageFont.truetype(font_bold_path, 18)
            self.font_small = ImageFont.truetype(font_path, 14)
            self.font_small_bold = ImageFont.truetype(font_bold_path, 14)
            self.font_live = ImageFont.truetype(font_path, 20)
            self.font_live_state = ImageFont.truetype(font_bold_path, 16)
        except IOError:
            fallback = ImageFont.load_default()
            self.font_header = fallback
            self.font_subheader = fallback
            self.font_body = fallback
            self.font_body_bold = fallback
            self.font_small = fallback
            self.font_small_bold = fallback
            self.font_live = fallback
            self.font_live_state = fallback

    # ──────────────────────────────────────────────────────────────────
    #  Main render
    # ──────────────────────────────────────────────────────────────────
    def render(self,
               camera_frame: Optional[cv2.Mat],
               session: ConversationSession,
               fps: float,
               scroll_offset: int = 0,
               hand_data: Optional[Dict] = None) -> cv2.Mat:
        """Renders the entire SAKSHAM UI and returns a BGR numpy frame."""

        # 1. Base canvas (OpenCV / numpy) for backgrounds & camera feed
        ui_frame = np.full((self.height, self.width, 3), config.COLOR_BG, dtype=np.uint8)

        # Header background
        cv2.rectangle(ui_frame, (0, 0), (self.width, self.header_h), (20, 20, 20), -1)
        cv2.line(ui_frame, (0, self.header_h), (self.width, self.header_h), config.COLOR_DIVIDER, 1)

        # Bottom section divider
        cv2.line(ui_frame, (0, self.bottom_y), (self.width, self.bottom_y), config.COLOR_DIVIDER, 1)

        # Vertical divider in bottom section (live transcript | camera+stats)
        cv2.line(ui_frame, (self.live_panel_w, self.bottom_y),
                 (self.live_panel_w, self.status_y), config.COLOR_DIVIDER, 1)

        # Status bar background
        cv2.rectangle(ui_frame, (0, self.status_y), (self.width, self.height), (20, 20, 20), -1)
        cv2.line(ui_frame, (0, self.status_y), (self.width, self.status_y), config.COLOR_DIVIDER, 1)

        # Live Transcript panel – slightly lighter background to differentiate
        cv2.rectangle(ui_frame, (0, self.bottom_y + 1),
                      (self.live_panel_w - 1, self.status_y - 1), (35, 35, 35), -1)

        # Camera preview (right panel, top-right)
        cam_x = self.right_panel_x + 20
        cam_y = self.bottom_y + 15
        if camera_frame is not None:
            cam_resized = cv2.resize(camera_frame, (self.cam_w, self.cam_h))
            ui_frame[cam_y:cam_y + self.cam_h, cam_x:cam_x + self.cam_w] = cam_resized
            cv2.rectangle(ui_frame, (cam_x, cam_y),
                          (cam_x + self.cam_w, cam_y + self.cam_h), config.COLOR_DIVIDER, 1)

            if config.DEV_MODE and hand_data and "pixel_landmarks" in hand_data:
                self._draw_dev_landmarks(ui_frame, hand_data["pixel_landmarks"], cam_x, cam_y)

        # 2. PIL overlay for all text rendering
        img_rgb = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        self._draw_header(draw)
        self._draw_conversation(draw, session, scroll_offset)
        self._draw_live_transcript(draw, session)
        self._draw_session_info(draw, session, cam_x, cam_y)
        self._draw_status_bar(draw, fps, session.mic_state, camera_frame is not None)
        self._draw_toast(draw, session)

        # Convert back to BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ──────────────────────────────────────────────────────────────────
    #  Header
    # ──────────────────────────────────────────────────────────────────
    def _draw_header(self, draw: ImageDraw.ImageDraw):
        draw.text((24, 12), "SAKSHAM V1",
                  font=self.font_header, fill=self._bgr_to_rgb(config.COLOR_ACCENT))
        draw.text((24, 40), "AI Communication Assistant",
                  font=self.font_subheader, fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

        hint = "[C] Clear  [E] Export  [O] Open Exports  [D] Dev Mode"
        _, _, w, _ = draw.textbbox((0, 0), hint, font=self.font_small)
        draw.text((self.width - w - 24, 24), hint,
                  font=self.font_small, fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

    # ──────────────────────────────────────────────────────────────────
    #  Conversation panel  (chat bubbles, scrollable)
    # ──────────────────────────────────────────────────────────────────
    def _draw_conversation(self, draw: ImageDraw.ImageDraw,
                           session: ConversationSession, scroll_offset: int):
        padding_x = 24
        base_y = self.bottom_y - 12          # bottom edge with margin
        avail_w = self.width - (padding_x * 2)
        max_bubble_w = int(avail_w * self.max_bubble_w_ratio)
        # How many chars fit – rough estimate for textwrap
        chars_per_line = max(30, int(max_bubble_w / 10))

        bubble_pad = 12
        bubble_gap = 10

        # Build layout bottom-up (newest at bottom)
        rendered: List[dict] = []
        for msg in reversed(session.messages):
            header = f"[{msg['time']}]  {msg['sender']}"
            wrapped = textwrap.fill(msg['text'], width=chars_per_line)
            _, _, hw, hh = draw.textbbox((0, 0), header, font=self.font_small)
            _, _, tw, th = draw.textbbox((0, 0), wrapped, font=self.font_body)
            bw = min(max(hw, tw) + bubble_pad * 2, max_bubble_w)
            bh = hh + th + bubble_pad * 2 + 4
            rendered.append({
                "header": header, "wrapped": wrapped,
                "w": bw, "h": bh
            })

        # Apply scroll (skip N newest messages)
        visible = rendered[scroll_offset:] if scroll_offset < len(rendered) else []

        cur_y = base_y
        for b in visible:
            cur_y -= b["h"]
            if cur_y < self.conv_y + 4:
                break

            # Bubble background
            draw.rounded_rectangle(
                [padding_x, cur_y, padding_x + b["w"], cur_y + b["h"]],
                radius=8,
                fill=(42, 42, 48, 255),
                outline=(70, 70, 75, 255)
            )
            # Header (timestamp + sender)
            draw.text((padding_x + bubble_pad, cur_y + bubble_pad),
                      b["header"], font=self.font_small,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))
            _, _, _, hh = draw.textbbox((0, 0), b["header"], font=self.font_small)
            # Body
            draw.text((padding_x + bubble_pad, cur_y + bubble_pad + hh + 4),
                      b["wrapped"], font=self.font_body,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))

            cur_y -= bubble_gap

        # Empty-state placeholder
        if not session.messages:
            hint = "No messages yet – start speaking to begin a conversation."
            _, _, tw, _ = draw.textbbox((0, 0), hint, font=self.font_body)
            cx = (self.width - tw) // 2
            cy = self.conv_y + self.conv_h // 2
            draw.text((cx, cy), hint, font=self.font_body,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

    # ──────────────────────────────────────────────────────────────────
    #  Live Transcript panel  (bottom-left, dedicated area)
    # ──────────────────────────────────────────────────────────────────
    def _draw_live_transcript(self, draw: ImageDraw.ImageDraw,
                              session: ConversationSession):
        px = 24
        py = self.bottom_y + 12

        # Panel title
        draw.text((px, py), "Live Transcript",
                  font=self.font_small_bold,
                  fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

        py += 24

        mic = session.mic_state
        draft = session.draft_message or ""

        if mic == "Listening":
            # Mic icon + state
            state_text = "\U0001f3a4  Listening..."
            state_color = self._bgr_to_rgb(config.COLOR_WARNING)
            draw.text((px, py), state_text, font=self.font_live_state, fill=state_color)
            py += 28

            # Show whatever partial/draft text we have
            if draft and draft != "...":
                wrapped = textwrap.fill(draft, width=70)
                draw.text((px, py), wrapped, font=self.font_live, fill=(220, 220, 220))
            else:
                draw.text((px, py), "Speak now...",
                          font=self.font_live, fill=(130, 130, 130))

        elif mic == "Processing":
            state_text = "\u23F3  Processing..."
            state_color = self._bgr_to_rgb(config.COLOR_SUCCESS)
            draw.text((px, py), state_text, font=self.font_live_state, fill=state_color)
            py += 28
            if draft:
                wrapped = textwrap.fill(draft, width=70)
                draw.text((px, py), wrapped, font=self.font_live, fill=(180, 180, 180))

        elif mic == "Error":
            draw.text((px, py), "\u274C  Microphone Error",
                      font=self.font_live_state, fill=(200, 60, 60))

        else:  # Idle
            draw.text((px, py), "\u23F8  Idle – Microphone standby",
                      font=self.font_live_state, fill=(120, 120, 120))
            py += 28
            draw.text((px, py), "Waiting for speech input...",
                      font=self.font_live, fill=(100, 100, 100))

    # ──────────────────────────────────────────────────────────────────
    #  Session Info panel  (bottom-right, below camera)
    # ──────────────────────────────────────────────────────────────────
    def _draw_session_info(self, draw: ImageDraw.ImageDraw,
                           session: ConversationSession,
                           cam_x: int, cam_y: int):
        sx = cam_x + self.cam_w + 24
        sy = self.bottom_y + 15

        draw.text((sx, sy), "Current Session",
                  font=self.font_small_bold,
                  fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))
        sy += 24

        info_lines = [
            f"Messages:  {session.get_message_count()}",
            f"Duration:  {session.get_duration_minutes()} min",
        ]

        # Export status
        if session.last_export_status == "Success" and session.last_export_path:
            info_lines.append(f"Last Export:  \u2713 {session.last_export_path}")
        elif session.last_export_status == "Failed":
            info_lines.append("Last Export:  \u2717 Failed")
        else:
            info_lines.append("Last Export:  \u2014")

        for line in info_lines:
            draw.text((sx, sy), line, font=self.font_small,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))
            sy += 22

    # ──────────────────────────────────────────────────────────────────
    #  Status bar
    # ──────────────────────────────────────────────────────────────────
    def _draw_status_bar(self, draw: ImageDraw.ImageDraw,
                         fps: float, mic_state: str, cam_ok: bool):
        ty = self.status_y + 9
        cx = 24

        # Mic state indicator
        mic_colors = {
            "Listening": config.COLOR_WARNING,
            "Processing": config.COLOR_SUCCESS,
            "Error": (0, 0, 200),
        }
        mic_color = mic_colors.get(mic_state, config.COLOR_TEXT_SECONDARY)
        draw.text((cx, ty), f"Mic: {mic_state}",
                  font=self.font_small, fill=self._bgr_to_rgb(mic_color))
        cx += 140

        # Camera
        cam_color = config.COLOR_SUCCESS if cam_ok else config.COLOR_WARNING
        draw.text((cx, ty), f"Camera: {'OK' if cam_ok else 'OFF'}",
                  font=self.font_small, fill=self._bgr_to_rgb(cam_color))
        cx += 110

        # Ready
        draw.text((cx, ty), "System Ready",
                  font=self.font_small, fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))

        # Dev Mode extras
        if config.DEV_MODE:
            draw.text((cx + 140, ty), f"DEV | FPS: {fps:.1f}",
                      font=self.font_small, fill=self._bgr_to_rgb(config.COLOR_WARNING))

    # ──────────────────────────────────────────────────────────────────
    #  Toast notification
    # ──────────────────────────────────────────────────────────────────
    def _draw_toast(self, draw: ImageDraw.ImageDraw, session: ConversationSession):
        if session.toast_message and time.time() < session.toast_expiry:
            msg = session.toast_message
            _, _, tw, th = draw.textbbox((0, 0), msg, font=self.font_body_bold)
            tx = (self.width - tw) // 2
            ty = self.header_h + 16
            draw.rounded_rectangle(
                [tx - 14, ty - 8, tx + tw + 14, ty + th + 8],
                radius=8, fill=(40, 160, 40, 220)
            )
            draw.text((tx, ty), msg, font=self.font_body_bold, fill=(255, 255, 255, 255))

    # ──────────────────────────────────────────────────────────────────
    #  Dev mode landmarks overlay (OpenCV, drawn before PIL pass)
    # ──────────────────────────────────────────────────────────────────
    def _draw_dev_landmarks(self, frame: cv2.Mat,
                            pixel_landmarks: List[Tuple[int, int]],
                            cx: int, cy: int):
        scale_x = self.cam_w / config.FRAME_WIDTH
        scale_y = self.cam_h / config.FRAME_HEIGHT
        for pt in pixel_landmarks:
            px = cx + int(pt[0] * scale_x)
            py = cy + int(pt[1] * scale_y)
            cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)

    # ──────────────────────────────────────────────────────────────────
    #  Utility
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _bgr_to_rgb(bgr_tuple: tuple) -> tuple:
        """Convert a BGR colour (used in config) to RGB for PIL."""
        return bgr_tuple[::-1]
