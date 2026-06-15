import cv2
import time
import logging
import sys

import src.config.settings as config
from src.camera.camera_manager import CameraManager
from src.sign_language.tracker import HandTracker
from src.speech.speech_engine import SpeechEngine
from src.ui.renderer import UIRenderer
from src.session_manager import ConversationSession

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("saksham.log", mode="w")
    ]
)

# Global state for OpenCV callbacks
scroll_offset = 0

def mouse_callback(event, x, y, flags, param):
    global scroll_offset
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scroll_offset += 1  # Scroll Up
        else:
            scroll_offset -= 1  # Scroll Down

def main():
    global scroll_offset
    
    logging.info("==========================================")
    logging.info("        SAKSHAM V1.5 STARTED              ")
    logging.info("==========================================")

    session = ConversationSession()

    # ── Module initialisation ──────────────────────────────────────────────
    camera = CameraManager(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT
    )
    if not camera.start():
        logging.critical("Failed to start the camera. Exiting.")
        sys.exit(1)

    tracker = HandTracker()
    ui = UIRenderer(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    
    # Initialize speech engine with callbacks
    def on_speech_text(text: str):
        session.add_message("Hearing Person", text)
        
    def on_speech_state(state: str):
        session.mic_state = state
        if state == "Listening":
            session.set_draft("...")
        elif state == "Processing":
            session.set_draft("Transcribing...")
        else:
            session.set_draft("")

    speech_engine = SpeechEngine(on_text_callback=on_speech_text, on_state_callback=on_speech_state)
    speech_engine.start()

    main_fps = 0.0
    ema_alpha = 0.1
    elapsed_time = 1.0 / config.TARGET_FPS

    window_name = "SAKSHAM V1 - AI Communication Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    logging.info("Main loop active. Press 'q' / ESC to quit, 'd' to toggle DEV MODE.")

    try:
        while True:
            t_start = time.perf_counter()

            # ── 1. Grab frame ──────────────────────────────────────────
            success, raw_frame = camera.get_frame()

            if not success or raw_frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(raw_frame, 1)

            # ── 2. Hand tracking (Dev Mode only right now) ──
            hand_detected, hand_data = tracker.process_frame(frame)

            # ── 3. Render UI ─────────────────────────────────────────
            annotated_frame = ui.render(
                camera_frame=frame,
                session=session,
                fps=main_fps,
                scroll_offset=scroll_offset,
                hand_data=hand_data if hand_detected else None
            )

            # ── 4. Display ───────────────────────────────────────────
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                logging.info("Quit key pressed.")
                break
            elif key == ord('d'):
                config.DEV_MODE = not config.DEV_MODE
                logging.info(f"Developer Mode: {'ON' if config.DEV_MODE else 'OFF'}")
            elif key == ord('c'):
                session.clear()
            elif key == ord('e'):
                session.export()
            elif key == ord('o'):
                # Open exports folder in Explorer (Windows)
                import subprocess, shlex
                try:
                    subprocess.Popen(['explorer', 'exports'], cwd='e:/gessture')
                    logging.info('Opened exports folder.')
                except Exception as e:
                    logging.error(f'Failed to open exports folder: {e}')

            # Keyboard scrolling fallback
            if key == 0:  # Up Arrow
                scroll_offset += 1
            elif key == 1:  # Down Arrow
                scroll_offset -= 1

            # Keep scroll offset from going below 0
            if scroll_offset < 0:
                scroll_offset = 0

            # ── 5. Frame-rate cap ────────────────────────────────────
            elapsed_time = time.perf_counter() - t_start
            sleep_needed = (1.0 / config.TARGET_FPS) - elapsed_time
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            t_end = time.perf_counter()
            loop_time = t_end - t_start
            if loop_time > 0:
                main_fps = ema_alpha * (1.0 / loop_time) + (1 - ema_alpha) * main_fps
            elapsed_time = loop_time if loop_time > 0 else (1.0 / config.TARGET_FPS)

    except Exception as e:
        logging.exception(f"Unhandled error: {e}")

    finally:
        camera.stop()
        speech_engine.stop()
        cv2.destroyAllWindows()
        logging.info("SAKSHAM V1.5 shutdown cleanly.")
        logging.info("==========================================")

if __name__ == "__main__":
    main()
