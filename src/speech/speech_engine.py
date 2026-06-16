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
        
        self.mic = sr.Microphone()
        self.stop_listening_func: Optional[Callable] = None
        self.mic_available = False

        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.mic_available = True
            logging.info("SpeechEngine: Microphone initialized successfully.")
        except Exception as e:
            logging.error(f"SpeechEngine: Microphone initialization failed: {e}")

        self.is_running = False
        self.is_paused = False
        self.last_recognized_text = ""
        self.last_recognized_time = 0.0
        self.duplicate_cooldown = 2.5

    def _audio_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        """Callback fired by SpeechRecognition background thread when a phrase completes."""
        if not self.is_running or self.is_paused:
            return
            
        try:
            logging.debug("SpeechEngine: Audio captured, processing...")
            self.update_state("Processing")
            text = recognizer.recognize_google(audio)
            
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
            logging.error(f"SpeechEngine: API request error: {e}")
        except Exception as e:
            logging.error(f"SpeechEngine: Unexpected error during recognition: {e}")
        finally:
            if self.is_running and not self.is_paused:
                self.update_state("Listening")

    def pause(self):
        """Temporarily stop processing audio (e.g., when TTS is speaking)."""
        self.is_paused = True
        self.update_state("Idle")

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
        logging.info("SpeechEngine: Starting background listening...")
        self.stop_listening_func = self.recognizer.listen_in_background(
            self.mic, 
            self._audio_callback,
            phrase_time_limit=10 
        )

    def stop(self):
        if not self.is_running:
            return
            
        logging.info("SpeechEngine: Stopping background listening...")
        self.is_running = False
        if self.stop_listening_func:
            self.stop_listening_func(wait_for_stop=False)
            self.stop_listening_func = None
        self.update_state("Idle")
