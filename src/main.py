import cv2
import time
import logging
import sys
import os

import src.config as config
from src.camera import CameraManager
from src.hand_tracker import HandTracker
from src.gesture_engine import GestureEngine
from src.ui import UIManager
from src.virtual_mouse import VirtualMouse
from src.whiteboard import Whiteboard


class AppState:
    """Manages the application's top-level mode."""
    def __init__(self):
        self.mode = "MOUSE"   # "MOUSE" | "WHITEBOARD"


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("antigravity.log", mode="w")
    ]
)


def main():
    logging.info("==========================================")
    logging.info("  ANTIGRAVITY GESTURE INTERFACE STARTED  ")
    logging.info("==========================================")

    # ── Module initialisation ──────────────────────────────────────────────
    camera = CameraManager(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT
    )
    if not camera.start():
        logging.critical("Failed to start the camera. Exiting.")
        sys.exit(1)

    tracker    = HandTracker()
    engine     = GestureEngine()
    ui         = UIManager()
    mouse      = VirtualMouse()
    whiteboard = Whiteboard(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    state      = AppState()

    # ── Mode switching helpers ─────────────────────────────────────────────
    def enter_whiteboard():
        state.mode = "MOUSE"   # will be flipped below via toggle
        toggle_whiteboard()

    def toggle_whiteboard():
        if state.mode == "MOUSE":
            state.mode = "WHITEBOARD"
            logging.info("Mode → WHITEBOARD")
            ui.trigger_toast("WHITEBOARD MODE ENABLED", duration=2.5)
        else:
            state.mode = "MOUSE"
            logging.info("Mode → MOUSE")
            ui.trigger_toast("WHITEBOARD MODE DISABLED", duration=2.5)

    # ── FPS / timing ──────────────────────────────────────────────────────
    main_fps   = 0.0
    ema_alpha  = 0.1
    elapsed_time = 1.0 / config.TARGET_FPS   # safe default

    # ── THUMBS_UP toggle state machine ────────────────────────────────────
    # HOLD-REQUIRED: gesture must be held for 1.5s continuously.
    # After firing, a 2s cooldown prevents re-triggering.
    # Progress (0.0-1.0) is rendered as an on-screen bar.
    THUMBS_HOLD_S   = 1.5          # seconds of continuous hold required
    THUMBS_COOLDOWN = 2.0          # post-toggle cooldown
    thumb_hold_start   = 0.0       # when the current hold began
    thumb_hold_active  = False     # True while gesture is being held
    thumb_hold_progress = 0.0      # 0.0-1.0 shown as progress bar
    last_thumbs_time   = 0.0       # cooldown reference

    # ── Profiling ─────────────────────────────────────────────────────────
    t_grab_sum = t_flip_sum = t_track_sum = 0.0
    t_gesture_sum = t_mouse_sum = t_hud_sum = t_show_sum = t_sleep_sum = 0.0
    profile_frame_count = 0

    window_name = "AntiGravity - Futuristic Gesture Control v1.0"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.info("Main loop active. Press 'q' / ESC to quit, 'h' to toggle HUD.")

    try:
        while True:
            t_start = time.perf_counter()

            # ── 1. Grab frame ──────────────────────────────────────────
            t0 = time.perf_counter()
            success, raw_frame = camera.get_frame()
            t_grab_sum += time.perf_counter() - t0

            if not success or raw_frame is None:
                time.sleep(0.005)
                continue

            # ── 2. Mirror ─────────────────────────────────────────────
            t0 = time.perf_counter()
            frame = cv2.flip(raw_frame, 1)
            t_flip_sum += time.perf_counter() - t0

            # ── 3. Composite whiteboard canvas onto feed ───────────────
            if state.mode == "WHITEBOARD":
                frame = whiteboard.get_overlay(frame)

            # ── 4. Hand tracking ──────────────────────────────────────
            t0 = time.perf_counter()
            hand_detected, hand_data = tracker.process_frame(frame)
            t_track_sum += time.perf_counter() - t0

            # ── 5. Gesture recognition ────────────────────────────────
            t0 = time.perf_counter()
            if hand_detected:
                active_gesture, metrics = engine.process_gestures(hand_data)
            else:
                active_gesture = "NONE"
                metrics = {
                    "confidence": 0.0,
                    "finger_states": {
                        "thumb": False, "index": False,
                        "middle": False, "ring": False, "pinky": False
                    },
                    "landmark_count": 0
                }
                hand_data = {}
            t_gesture_sum += time.perf_counter() - t0

            # ── 6. THUMBS_UP toggle (HOLD-REQUIRED, 1.5 s + 2 s cooldown) ──
            now_wall = time.time()
            in_cooldown = (now_wall - last_thumbs_time) < THUMBS_COOLDOWN

            if active_gesture == "THUMBS_UP" and not in_cooldown:
                if not thumb_hold_active:
                    # Gesture just started — begin timing
                    thumb_hold_active  = True
                    thumb_hold_start   = now_wall
                    thumb_hold_progress = 0.0
                else:
                    held = now_wall - thumb_hold_start
                    thumb_hold_progress = min(1.0, held / THUMBS_HOLD_S)
                    if held >= THUMBS_HOLD_S:
                        toggle_whiteboard()
                        last_thumbs_time   = now_wall
                        thumb_hold_active  = False
                        thumb_hold_progress = 0.0
            else:
                # Gesture broken or in cooldown — reset
                thumb_hold_active   = False
                thumb_hold_progress = 0.0

            # ── 7. Whiteboard logic ───────────────────────────────────
            if state.mode == "WHITEBOARD" and hand_detected:
                lms = hand_data["pixel_landmarks"]
                # Use weighted average of index tip [8] and PIP joint [7]
                # for a more stable drawing anchor than raw tip alone.
                raw_tip = lms[8]
                pip_jnt = lms[7]
                stable_pt = (
                    (raw_tip[0] * 2 + pip_jnt[0]) // 3,
                    (raw_tip[1] * 2 + pip_jnt[1]) // 3
                )

                # Toolbar hover uses the stable point
                wb_toast = whiteboard.update_toolbar(stable_pt, elapsed_time)
                if wb_toast:
                    ui.trigger_toast(wb_toast)

                # Drawing (PINCH = draw) — pass stable point
                whiteboard.process_drawing(stable_pt, active_gesture)

            # ── 8. Virtual mouse ──────────────────────────────────────
            t0 = time.perf_counter()
            # In WHITEBOARD mode suppress real mouse events so we don't
            # accidentally click background applications.
            mouse_gesture = active_gesture if state.mode == "MOUSE" else "NONE"
            cursor_x, cursor_y, is_clicked = mouse.update(hand_data, mouse_gesture)
            cursor_pos = (cursor_x, cursor_y)
            t_mouse_sum += time.perf_counter() - t0

            # ── 9. Render HUD ─────────────────────────────────────────
            t0 = time.perf_counter()

            index_tip_for_wb = (
                hand_data["pixel_landmarks"][8]
                if (state.mode == "WHITEBOARD" and hand_detected)
                else None
            )

            annotated_frame = ui.draw_hud(
                frame=frame,
                hand_detected=hand_detected,
                hand_data=hand_data,
                active_gesture=active_gesture,
                gesture_metrics=metrics,
                fps=main_fps,
                cursor_pos=cursor_pos,
                is_clicked=is_clicked,
                mouse_diagnostics=mouse.get_diagnostics(),
                active_mode=state.mode,
                whiteboard=whiteboard if state.mode == "WHITEBOARD" else None,
                index_tip=index_tip_for_wb,
                thumb_hold_progress=thumb_hold_progress,
                dt=elapsed_time
            )
            t_hud_sum += time.perf_counter() - t0

            # ── 10. Display ───────────────────────────────────────────
            t0 = time.perf_counter()
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            t_show_sum += time.perf_counter() - t0

            if key in (ord('q'), 27):
                logging.info("Quit key pressed.")
                break
            elif key == ord('h'):
                config.DISABLE_HUD_EFFECTS = not config.DISABLE_HUD_EFFECTS

            # ── 11. Frame-rate cap ────────────────────────────────────
            t0 = time.perf_counter()
            elapsed_time = time.perf_counter() - t_start
            sleep_needed = (1.0 / config.TARGET_FPS) - elapsed_time
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            t_sleep_sum += time.perf_counter() - t0

            # FPS (EMA)
            t_end     = time.perf_counter()
            loop_time = t_end - t_start
            if loop_time > 0:
                main_fps = ema_alpha * (1.0 / loop_time) + (1 - ema_alpha) * main_fps
            elapsed_time = loop_time if loop_time > 0 else (1.0 / config.TARGET_FPS)

            # ── 12. Profiling ─────────────────────────────────────────
            profile_frame_count += 1
            if profile_frame_count >= 100:
                logging.info(
                    f"\n--- PIPELINE PROFILING REPORT (avg ms/frame over 100 frames) ---\n"
                    f"Frame Grab:     {t_grab_sum * 10:.2f} ms\n"
                    f"Frame Flip:     {t_flip_sum * 10:.2f} ms\n"
                    f"Hand Tracking:  {t_track_sum * 10:.2f} ms\n"
                    f"Gesture Engine: {t_gesture_sum * 10:.2f} ms\n"
                    f"Virtual Mouse:  {t_mouse_sum * 10:.2f} ms\n"
                    f"HUD Drawing:    {t_hud_sum * 10:.2f} ms\n"
                    f"Window Display: {t_show_sum * 10:.2f} ms\n"
                    f"Sleep Delay:    {t_sleep_sum * 10:.2f} ms\n"
                    f"Estimated FPS:  {main_fps:.1f}\n"
                    f"------------------------------------------------------------------"
                )
                t_grab_sum = t_flip_sum = t_track_sum = t_gesture_sum = 0.0
                t_mouse_sum = t_hud_sum = t_show_sum = t_sleep_sum = 0.0
                profile_frame_count = 0

    except Exception as e:
        logging.exception(f"Unhandled error: {e}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logging.info("AntiGravity interface shutdown cleanly.")
        logging.info("==========================================")


if __name__ == "__main__":
    main()
