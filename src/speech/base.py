from typing import Callable

class BaseSpeechEngine:
    """Base class for all speech engines."""
    def __init__(self, on_text_callback: Callable[[str], None], on_state_callback: Callable[[str], None] = None):
        self.on_text_callback = on_text_callback
        self.on_state_callback = on_state_callback
        self.state = "Idle"

    def start(self):
        pass

    def stop(self):
        pass

    def update_state(self, new_state: str):
        if self.state != new_state:
            self.state = new_state
            if self.on_state_callback:
                self.on_state_callback(new_state)
