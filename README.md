# AntiGravity

AntiGravity is a futuristic, touchless computer control interface powered by hand tracking via a webcam.

## Features

- **Virtual Mouse Navigation:** Control your computer cursor with hand gestures. Pinch to click, double-click, and drag.
- **Gesture Control:** Uses a robust MediaPipe hand tracking pipeline to recognize gestures like OPEN_PALM, CLOSED_FIST, PINCH, POINT, THUMBS_UP, PEACE_SIGN, etc.
- **Cyberpunk HUD:** An overlay interface displaying tracking confidence, active gestures, and system metrics.
- **Virtual Whiteboard (Phase 3):** A gesture-controlled drawing canvas. Hold THUMBS_UP to enter whiteboard mode, draw with PINCH, and select tools (colors, brush sizes, eraser) using a hover-based side toolbar.

## Architecture

- **`main.py`**: The application loop and state machine.
- **`camera.py`**: Multi-threaded webcam capture.
- **`hand_tracker.py`**: MediaPipe integration for hand landmarks and finger state detection.
- **`gesture_engine.py`**: Real-time gesture recognition with temporal smoothing.
- **`virtual_mouse.py`**: Translates hand coordinates and gestures to Win32 mouse events.
- **`ui.py`**: OpenCV-based rendering for the HUD, hover buttons, and debug dashboard.
- **`whiteboard.py`**: Manages the drawing canvas, stroke smoothing, sub-pixel interpolation, and the hover toolbar.

## Requirements

- Python 3.9+
- Windows OS (for `win32api` virtual mouse)
- Webcam

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main
```
