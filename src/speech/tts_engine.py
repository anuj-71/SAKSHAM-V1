import threading
import queue
import logging
import pythoncom

try:
    import pyttsx3
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseTTSEngine:
    """Base class for Text-to-Speech engines."""
    def start(self):
        pass
    def stop(self):
        pass
    def speak(self, text: str):
        pass

class PyTTSx3Engine(BaseTTSEngine):
    """
    Offline Text-to-Speech Engine using pyttsx3.
    Runs on a background thread to prevent blocking the UI loop.
    """
    def __init__(self, on_speech_start=None, on_speech_end=None):
        self.speech_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.thread.start()
        logging.info("PyTTSx3Engine started.")

    def stop(self):
        self.is_running = False
        # Push a dummy item to unblock the queue
        self.speech_queue.put(None)
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info("PyTTSx3Engine stopped.")

    def speak(self, text: str):
        """Queues text to be spoken."""
        if text and text.strip():
            logging.info(f"TTS QUEUED: '{text}'")
            self.speech_queue.put(text.strip())

    def _tts_loop(self):
        # pyttsx3 engine must be initialized in the same thread it is used
        try:
            pythoncom.CoInitialize()
        except Exception as e:
            logging.warning(f"Could not initialize COM: {e}")

        try:
            engine = pyttsx3.init()
            # Optional: configure voice/rate here
            engine.setProperty('rate', 150) # Slower, clearer speech
        except Exception as e:
            logging.error(f"Failed to initialize pyttsx3 engine: {e}")
            return

        while self.is_running:
            try:
                # Block until there is text to speak
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    continue
                    
                logging.info(f"TTS STARTED: '{text}'")
                
                if self.on_speech_start:
                    self.on_speech_start()
                    
                engine.say(text)
                engine.runAndWait()
                
                if self.on_speech_end:
                    self.on_speech_end()

                logging.info(f"TTS FINISHED: '{text}'")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS FAILED: {e}")
