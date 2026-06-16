import time
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import logging

class ConversationSession:
    """
    Manages the state and history of a SAKSHAM communication session.
    """
    def __init__(self):
        self.messages: List[Dict] = []
        self.start_time = time.time()
        self.draft_message = ""
        self.mic_state = "Idle" # Idle, Listening, Processing
        self.toast_message: Optional[str] = None
        self.toast_expiry: float = 0.0
        self.last_export_path = ""
        self.last_export_status = ""  # "", "Success", "Failed"
        self.typing_buffer = ""
        self.is_typing_focused = False
        self.request_listen = False
        self.last_error = ""
        
        # Ensure export directory exists
        os.makedirs("exports", exist_ok=True)

    def add_message(self, sender: str, text: str, source: str = "System"):
        if not text.strip():
            return
            
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        self.messages.append({
            "sender": sender,
            "text": text,
            "time": time_str,
            "source": source
        })
        logging.info(f"[{time_str}] {sender} ({source}): {text}")
        
        # Clear draft when a final message is committed
        self.draft_message = ""

    def clear(self):
        """Clears the current conversation but keeps the session start time."""
        self.messages.clear()
        self.draft_message = ""
        self.set_toast("Conversation Cleared")
        logging.info("Conversation cleared.")

    def new_session(self):
        """Resets the entire session."""
        self.messages.clear()
        self.draft_message = ""
        self.start_time = time.time()
        self.set_toast("New Session Started")
        logging.info("New session started.")

    def get_duration_minutes(self) -> int:
        return int((time.time() - self.start_time) / 60)
        
    def get_message_count(self) -> int:
        return len(self.messages)

    def set_draft(self, text: str):
        self.draft_message = text

    def set_toast(self, message: str, duration: float = 3.0):
        self.toast_message = message
        self.toast_expiry = time.time() + duration

    def export(self) -> bool:
        """Exports the conversation to TXT and JSON in the exports/ directory."""
        if not self.messages:
            self.last_export_status = "Failed"
            self.set_toast("\u2717 Export Failed: Empty Chat")
            logging.warning("Export requested but conversation is empty.")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = f"exports/chat_{timestamp}.txt"
        json_path = f"exports/chat_{timestamp}.json"
        
        try:
            # TXT Export
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"SAKSHAM V1 Conversation Export\n\n")
                for msg in self.messages:
                    f.write(f"[{msg['time']}] {msg['sender']}\n{msg['text']}\n\n")

            # JSON Export
            export_data = {
                "session_start": datetime.fromtimestamp(self.start_time).isoformat(),
                "session_duration_minutes": self.get_duration_minutes(),
                "message_count": self.get_message_count(),
                "messages": self.messages
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)
                
            # Verify Export Success
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 and \
               os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                self.last_export_path = f"chat_{timestamp}"
                self.last_export_status = "Success"
                self.set_toast(f"\u2713 Exported: chat_{timestamp} ({self.get_message_count()} msgs)", duration=4.0)
                logging.info(f"Session exported successfully to {txt_path} and {json_path}")
                return True
            else:
                self.last_export_status = "Failed"
                self.set_toast("\u2717 Export Failed: Verification Error")
                return False
                
        except Exception as e:
            self.last_export_status = "Failed"
            self.set_toast("\u2717 Export Failed: System Error")
            logging.error(f"Export failed with exception: {e}")
            return False
