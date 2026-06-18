import threading
import speech_recognition as sr
import logging
import time
from typing import Callable, Optional

from src.speech.base import BaseSpeechEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpeechEngine(BaseSpeechEngine):
    """
    Phase 1 Speech Engine using SpeechRecognition and Google Web API.
    Runs reliably in a background thread.
    """
    def __init__(self, on_text_callback: Callable[[str], None], on_state_callback: Callable[[str], None] = None):
        super().__init__(on_text_callback, on_state_callback)
        self.recognizer = sr.Recognizer()
        
        # Optimize for ambient noise and latency
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # Defer Microphone creation into a safe block — PyAudio may be missing.
        self.mic = None
        self.stop_listening_func: Optional[Callable] = None
        self.mic_available = False

        # Manual listen state (used by UI when Listen button pressed)
        self.is_manual_listening = False
        self.last_error = None

        try:
            # Attempt to instantiate the Microphone (this can raise if PyAudio is missing)
            self.mic = sr.Microphone()
            try:
                with self.mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.mic_available = True
                logging.info("SpeechEngine: Microphone initialized successfully.")
            except Exception as inner_e:
                logging.exception(f"SpeechEngine: Microphone present but failed during ambient adjust: {inner_e}")
                self.last_error = str(inner_e)
                self.mic_available = False
        except Exception as e:
            # Common failure: PyAudio not installed or no audio drivers available.
            logging.warning(f"SpeechEngine: Could not initialize Microphone (PyAudio may be missing): {e}")
            self.last_error = "Could not find PyAudio; check installation"
            self.mic = None
            self.mic_available = False

        self.is_running = False
        self.is_paused = False
        self.last_recognized_text = ""
        self.last_recognized_time = 0.0
        self.duplicate_cooldown = 2.5

    def _process_audio(self, audio: sr.AudioData):
        """Processes audio in a worker thread to prevent blocking the listener."""
        def worker():
            try:
                logging.debug("SpeechEngine: Audio captured, processing...")
                if not self.is_manual_listening:
                    self.update_state("Processing")
                    
                text = self.recognizer.recognize_google(audio)
                
                # Additional check to prevent TTS audio feedback if pause was slightly delayed
                if self.is_paused:
                    logging.debug("SpeechEngine: Ignored speech because engine is paused.")
                    return
                    
                if text:
                    text = text.strip()
                    if text:
                        current_time = time.time()
                        if text == self.last_recognized_text and (current_time - self.last_recognized_time) < self.duplicate_cooldown:
                            logging.info(f"SpeechEngine: Ignored duplicate speech within cooldown: '{text}'")
                        else:
                            self.last_recognized_text = text
                            self.last_recognized_time = current_time
                            logging.info(f"SpeechEngine: Speech recognized successfully: '{text}'")
                            self.on_text_callback(text)
                            logging.debug("SpeechEngine: Callback executed successfully.")
            except sr.UnknownValueError:
                pass # Unintelligible audio, ignore
            except sr.RequestError as e:
                logging.exception(f"SpeechEngine: API request error: {e}")
                self.last_error = str(e)
            except Exception as e:
                logging.exception(f"SpeechEngine: Unexpected error during recognition: {e}")
                self.last_error = str(e)
            finally:
                if self.is_running and not self.is_paused and not self.is_manual_listening:
                    self.update_state("Listening")

        threading.Thread(target=worker, daemon=True).start()

    def manual_listen(self, timeout: float = 6.0, phrase_time_limit: float = 10.0):
        """Performs a single-shot synchronous listen and recognition.

        Intended to be called from a worker thread so it doesn't block the UI.
        Adds result via the same duplicate-cooldown logic as the background callback.
        """
        if not self.mic_available:
            logging.error("SpeechEngine.manual_listen: Microphone unavailable.")
            self.last_error = "Microphone unavailable"
            self.update_state("Error")
            return

        if self.is_manual_listening:
            logging.info("SpeechEngine.manual_listen: Already listening (manual).")
            return

        self.is_manual_listening = True
        try:
            with self.mic as source:
                logging.info("SpeechEngine.manual_listen: Listening (manual)...")
                # Optional short ambient adjust before manual listen
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                except Exception:
                    pass
                self.update_state("Processing")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            try:
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                logging.info("SpeechEngine.manual_listen: Unintelligible audio.")
                return
            except sr.RequestError as e:
                logging.exception(f"SpeechEngine.manual_listen: API request error: {e}")
                self.last_error = str(e)
                self.update_state("Error")
                return

            if text:
                text = text.strip()
                if text:
                    current_time = time.time()
                    if text == self.last_recognized_text and (current_time - self.last_recognized_time) < self.duplicate_cooldown:
                        logging.info(f"SpeechEngine.manual_listen: Ignored duplicate within cooldown: '{text}'")
                    else:
                        self.last_recognized_text = text
                        self.last_recognized_time = current_time
                        logging.info(f"SpeechEngine.manual_listen: Recognized: '{text}'")
                        try:
                            self.on_text_callback(text)
                        except Exception:
                            logging.exception("SpeechEngine.manual_listen: on_text_callback raised an exception")
        finally:
            self.is_manual_listening = False
            if self.is_running and not self.is_paused:
                self.update_state("Listening")

    def pause(self):
        """Temporarily stop processing audio (e.g., when TTS is speaking)."""
        self.is_paused = True
        self.update_state("Mic Off")

    def resume(self):
        """Resume processing audio."""
        self.is_paused = False
        # Reset last text so it can be recognized again if spoken intentionally later
        self.last_recognized_text = "" 
        if self.is_running:
            self.update_state("Listening")

    def start(self):
        if not self.mic_available:
            self.update_state("Error")
            logging.error("SpeechEngine: Cannot start, microphone unavailable.")
            return

        if self.is_running:
            return

        self.is_running = True
        self.update_state("Listening")
        logging.info("SpeechEngine: Starting manual background listening loop...")
        
        self.stop_listening_func = True  # Used as a dummy flag for UI
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _listen_loop(self):
        try:
            with self.mic as source:
                while self.is_running:
                    if self.is_paused:
                        self.update_state("Idle")
                        time.sleep(0.1)
                        continue

                    if not self.is_manual_listening:
                        self.update_state("Listening")
                        
                    try:
                        audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=8.0)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        if self.is_running:
                            logging.exception(f"SpeechEngine: Error capturing audio: {e}")
                        time.sleep(1)
                        continue

                    # If paused during recording, discard
                    if self.is_paused or not self.is_running:
                        continue

                    self._process_audio(audio)
        except Exception as e:
            logging.exception(f"SpeechEngine: Listener loop crashed: {e}")

    def stop(self):
        if not self.is_running:
            self.update_state("Mic Off")
            return
            
        logging.info("SpeechEngine: Stopping background listening...")
        self.is_running = False
        self.stop_listening_func = None
        self.update_state("Mic Off")
