import cv2
import time
import logging
import sys

import src.config.settings as config
from src.camera.camera_manager import CameraManager
from src.sign_language.engine import SignLanguageEngine
from src.speech.speech_engine import SpeechEngine
from src.speech.tts_engine import PyTTSx3Engine
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
global_session = None

def mouse_callback(event, x, y, flags, param):
    global scroll_offset, global_session
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scroll_offset += 1  # Scroll Up
        else:
            scroll_offset -= 1  # Scroll Down
    elif event == cv2.EVENT_LBUTTONDOWN:
        if global_session:
            # Check if clicked in Input Bar (height - 86 to height - 36)
            h = config.FRAME_HEIGHT
            w = config.FRAME_WIDTH
            if h - 86 <= y <= h - 36:
                mic_x1 = w - 210
                mic_x2 = w - 110
                if x >= w - 100:
                    if global_session.typing_buffer.strip():
                        global_session.add_message("Hearing Person", global_session.typing_buffer.strip(), source="Typed")
                    global_session.typing_buffer = ""
                    global_session.is_typing_focused = False
                elif mic_x1 <= x <= mic_x2:
                    global_session.request_mic_toggle = True
                    global_session.is_typing_focused = False
                else:
                    global_session.is_typing_focused = True
            else:
                global_session.is_typing_focused = False

def main():
    global scroll_offset
    
    logging.info("==========================================")
    logging.info("        SAKSHAM V2.0 STARTED              ")
    logging.info("==========================================")

    session = ConversationSession()
    global global_session
    global_session = session

    # ── Module initialisation ──────────────────────────────────────────────
    camera = CameraManager(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT
    )
    if not camera.start():
        logging.critical("Failed to start the camera. Exiting.")
        sys.exit(1)

    ui = UIRenderer(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    
    # ── Text-to-Speech Engine ──
    def on_tts_start():
        logging.info("TTS started speaking, pausing STT mic.")
        if session.mic_enabled:
            speech_engine.pause()
        
    def on_tts_end():
        logging.info("TTS finished speaking, resuming STT mic.")
        if session.mic_enabled:
            speech_engine.resume()
        
    tts_engine = PyTTSx3Engine(on_speech_start=on_tts_start, on_speech_end=on_tts_end)
    tts_engine.start()

    # ── Sign Language Engine ──
    SIGN_TO_PHRASE = {
        "HELLO": "Hello",
        "HELP": "I need help",
        "WATER": "I need water",
        "THANK YOU": "Thank you",
        "STOP": "Please stop",
        "YES": "Yes",
        "NO": "No"
    }

    def on_sign_recognized(raw_sign_label: str):
        if session.dataset_mode:
            return
        # 1. Add RAW sign label to the conversation history
        session.add_message("Deaf User", raw_sign_label, source="Sign")
        
        # 2. Map to a natural phrase and speak it
        spoken_phrase = SIGN_TO_PHRASE.get(raw_sign_label, raw_sign_label)
        tts_engine.speak(spoken_phrase)

    sl_engine = SignLanguageEngine(on_sign_recognized=on_sign_recognized)
    
    # ── Speech-to-Text Engine ──
    def on_speech_text(text: str):
        session.add_message("Hearing Person", text, source="Speech")
        
    def on_speech_state(state: str):
        session.mic_state = state
        if state == "Listening":
            if not session.draft_message:
                session.set_draft("...")
        elif state == "Processing":
            session.set_draft("Transcribing...")
        else:
            session.set_draft("")

    speech_engine = SpeechEngine(on_text_callback=on_speech_text, on_state_callback=on_speech_state)

    def toggle_mic():
        if session.mic_enabled:
            speech_engine.stop()
            session.mic_enabled = False
            session.mic_state = "Mic Off"
            session.set_draft("")
            session.set_toast("Microphone Off")
            return

        speech_engine.start()
        if speech_engine.is_running:
            session.mic_enabled = True
            session.set_toast("Microphone On")
        else:
            session.mic_enabled = False
            session.mic_state = "Error" if speech_engine.last_error else "Mic Off"
            if speech_engine.last_error:
                session.last_error = speech_engine.last_error
                session.set_toast(f"Mic: {speech_engine.last_error}")

    def start_dataset_recording():
        if sl_engine.has_review_clip():
            session.set_toast("Review or reject the pending clip first")
            return
        sl_engine.start_recording(session.current_dataset_label, session.current_signer_id)
        session.set_dataset_status("Recording")
        session.set_dataset_review_summary(None)
        session.set_toast(f"Recording {session.current_dataset_label} for {session.current_signer_id}")

    def stop_dataset_recording():
        review = sl_engine.stop_recording()
        session.set_dataset_review_summary(review)
        session.set_dataset_status("Review" if review else "Idle")
        if review:
            if not review["passes_quality_checks"]:
                session.set_toast("Clip blocked: fix sample quality before saving", duration=4.0)
            elif review["quality_warnings"]:
                session.set_toast("Clip has quality warnings; review before saving", duration=4.0)
            else:
                session.set_toast(f"Clip ready for review: {review['frame_count']} frames")

    def accept_dataset_clip():
        saved = sl_engine.accept_recording()
        if not saved:
            return
        if saved.get("status") == "blocked":
            blockers = saved.get("quality_blockers", [])
            blocker_text = blockers[0] if blockers else "Clip does not meet dataset quality thresholds."
            session.set_dataset_review_summary(sl_engine.get_review_summary())
            session.set_dataset_status("Review")
            session.set_toast(blocker_text, duration=4.0)
            return
        session.mark_dataset_clip_saved(saved["label"])
        session.set_dataset_review_summary(None)
        session.set_dataset_status("Idle")
        session.set_toast(f"Saved clip: {saved['label']} ({saved['frame_count']} frames)")

    def reject_dataset_clip(reason: str = "Rejected"):
        rejected = sl_engine.reject_recording(reason)
        if not rejected:
            return
        session.set_dataset_review_summary(None)
        session.set_dataset_status("Idle")
        session.set_toast(f"Clip discarded: {rejected['label']}")

    def toggle_dataset_mode():
        if session.dataset_mode:
            if sl_engine.dataset_manager.is_recording:
                stop_dataset_recording()
                reject_dataset_clip("Cancelled when exiting dataset mode")
            elif sl_engine.has_review_clip():
                reject_dataset_clip("Cancelled when exiting dataset mode")
            session.set_dataset_mode(False)
            return

        session.set_dataset_mode(True)
        session.set_dataset_status("Idle")
        session.set_dataset_review_summary(None)

    def export_dataset_summary():
        exported_files = sl_engine.export_dataset("both")
        if not exported_files:
            session.set_dataset_export_result("", False)
            session.set_toast("No accepted dataset clips available to export", duration=4.0)
            return

        export_base = exported_files[0].rsplit(".", 1)[0]
        session.set_dataset_export_result(export_base, True)
        session.set_toast(f"Dataset exported: {len(exported_files)} files", duration=4.0)

    main_fps = 0.0
    ema_alpha = 0.1
    elapsed_time = 1.0 / config.TARGET_FPS

    window_name = "SAKSHAM V2 - AI Communication Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    logging.info("Main loop active. Press 'q' / ESC to quit, 'd' to toggle DEV MODE.")

    try:
        while True:
            t_start = time.perf_counter()

            if session.request_mic_toggle:
                session.request_mic_toggle = False
                toggle_mic()

            # ── 1. Grab frame ──────────────────────────────────────────
            success, raw_frame = camera.get_frame()

            if not success or raw_frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(raw_frame, 1)

            # ── 2. Sign Language Tracking & Recognition ──
            hand_detected, hand_data = sl_engine.process_frame(frame)

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

            if key != 255: # A key was pressed
                if session.is_typing_focused:
                    if key == 13: # Enter
                        if session.typing_buffer.strip():
                            session.add_message("Hearing Person", session.typing_buffer.strip(), source="Typed")
                        session.typing_buffer = ""
                    elif key == 27: # Esc
                        session.is_typing_focused = False
                        session.typing_buffer = ""
                    elif key == 8: # Backspace
                        session.typing_buffer = session.typing_buffer[:-1]
                    elif 32 <= key <= 126: # Printable ASCII
                        session.typing_buffer += chr(key)
                else:
                    if key in (ord('q'), ord('Q'), 27):
                        logging.info("Quit key pressed.")
                        break
                    elif key in (ord('m'), ord('M')):
                        toggle_mic()
                    elif key in (ord('k'), ord('K')):
                        toggle_dataset_mode()
                    elif key in (ord('d'), ord('D')):
                        config.DEV_MODE = not config.DEV_MODE
                        logging.info(f"Developer Mode: {'ON' if config.DEV_MODE else 'OFF'}")
                    elif key in (ord('c'), ord('C')):
                        session.clear()
                    elif key in (ord('e'), ord('E')):
                        if session.dataset_mode:
                            export_dataset_summary()
                        else:
                            session.export()
                    elif key in (ord('o'), ord('O')):
                        # Open exports folder in Explorer (Windows)
                        import subprocess
                        try:
                            subprocess.Popen('explorer exports', shell=True, cwd='e:/gessture')
                            logging.info('Opened exports folder.')
                        except Exception as e:
                            logging.error(f'Failed to open exports folder: {e}')
                    elif session.dataset_mode and key in (ord('j'), ord('J')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_label(-1)
                    elif session.dataset_mode and key in (ord('l'), ord('L')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_label(1)
                    elif session.dataset_mode and key in (ord('u'), ord('U')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_signer(-1)
                    elif session.dataset_mode and key in (ord('i'), ord('I')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_signer(1)
                    elif session.dataset_mode and key in (ord('r'), ord('R')):
                        if sl_engine.dataset_manager.is_recording:
                            stop_dataset_recording()
                        elif sl_engine.has_review_clip():
                            reject_dataset_clip("Re-record requested")
                            start_dataset_recording()
                        else:
                            start_dataset_recording()
                    elif session.dataset_mode and key in (ord('a'), ord('A')):
                        accept_dataset_clip()
                    elif session.dataset_mode and key in (ord('x'), ord('X')):
                        reject_dataset_clip()

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
        tts_engine.stop()
        cv2.destroyAllWindows()
        logging.info("SAKSHAM V2.0 shutdown cleanly.")
        logging.info("==========================================")

if __name__ == "__main__":
    main()
