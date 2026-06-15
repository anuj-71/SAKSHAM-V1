# SAKSHAM-V1

SAKSHAM (Smart AI Knowledge Assistant for Sign, Hearing, and Accessible Multimodal Communication) is an AI-powered accessibility platform designed to bridge communication gaps between deaf, hard-of-hearing, and hearing individuals through speech recognition, sign language integration, and real-time communication assistance.

## Features

- **Accessibility-Focused UI:** Responsive chat bubble layout displaying conversation history with precise timestamps.
- **Microphone Integration:** Real-time state indicators (Listening, Processing, Idle) and a dedicated Live Transcript panel for speech-to-text.
- **Webcam Preview:** Clean camera preview for real-time video feedback.
- **Session Management:** Tracks message counts, session duration, and export history. Includes hotkeys to clear session or start a new one.
- **Conversation Export:** Robust export options (TXT and JSON) to the `exports/` folder with file verification.
- **Developer Mode:** Toggle dev overlays (FPS and tracking landmarks) using the `D` key.
- **Future Ready:** Modular design ready for Silero VAD + Faster-Whisper migration.

## Requirements

- Python 3.9+
- Windows OS (recommended for UI and audio APIs)
- Webcam
- Microphone

## Installation

1. Set up a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the main application loop:
```bash
python src/main.py
```

### Shortcuts
- `C` - Clear conversation
- `E` - Export conversation (exports both TXT and JSON to `exports/`)
- `O` - Open exports directory in File Explorer
- `D` - Toggle Developer Mode (FPS & Landmarks)
- `Q` / `ESC` - Quit application
