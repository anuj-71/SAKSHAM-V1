import threading
import speech_recognition as sr
import logging
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseSpeechEngine:
    """
    Abstract base class for speech-to-text engines.
    """
    def __init__(self, on_text_callback: Callable[[str], None], on_state_callback: Callable[[str], None] = None):
        self.on_text_callback = on_text_callback
        self.on_state_callback = on_state_callback
        self.is_running = False

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def update_state(self, state: str):
        if self.on_state_callback:
            self.on_state_callback(state)


class V1SpeechEngine(BaseSpeechEngine):
    """
    Phase 1 Speech Engine using SpeechRecognition and Google Web API.
    """
    def __init__(self, on_text_callback: Callable[[str], None], on_state_callback: Callable[[str], None] = None):
        super().__init__(on_text_callback, on_state_callback)
        self.recognizer = sr.Recognizer()
        
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
            logging.info("V1SpeechEngine: Microphone initialized successfully.")
        except Exception as e:
            logging.error(f"V1SpeechEngine: Microphone initialization failed: {e}")

    def _audio_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        """Callback fired by SpeechRecognition background thread when a phrase completes."""
        try:
            self.update_state("Processing")
            text = recognizer.recognize_google(audio)
            if text:
                self.on_text_callback(text)
        except sr.UnknownValueError:
            pass # Unintelligible audio, ignore
        except sr.RequestError as e:
            logging.error(f"V1SpeechEngine: API request error: {e}")
        except Exception as e:
            logging.error(f"V1SpeechEngine: Unexpected error during recognition: {e}")
        finally:
            self.update_state("Listening")

    def start(self):
        if not self.mic_available:
            self.update_state("Error")
            logging.error("V1SpeechEngine: Cannot start, microphone unavailable.")
            return

        if self.is_running:
            return

        self.is_running = True
        self.update_state("Listening")
        logging.info("V1SpeechEngine: Starting background listening...")
        self.stop_listening_func = self.recognizer.listen_in_background(
            self.mic, 
            self._audio_callback,
            phrase_time_limit=10 
        )

    def stop(self):
        if not self.is_running:
            return
            
        logging.info("V1SpeechEngine: Stopping background listening...")
        if self.stop_listening_func:
            self.stop_listening_func(wait_for_stop=False)
            self.stop_listening_func = None
        self.is_running = False
        self.update_state("Idle")

# Provide a standard export name so main.py doesn't care which engine is used
SpeechEngine = V1SpeechEngine
