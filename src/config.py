import cv2

# ==========================================
# ANTIGRAVITY CONFIGURATION SETTINGS
# ==========================================

# Camera Settings
CAMERA_INDEX = 0          # Default system camera
FRAME_WIDTH = 1280        # Camera capture width
FRAME_HEIGHT = 720        # Camera capture height
TARGET_FPS = 30           # Limit frame processing rate to save CPU

# Performance & Tracking Optimizations
TRACKING_FRAME_WIDTH = 480   # Resized width for MediaPipe processing (lower = faster)
TRACKING_FRAME_HEIGHT = 270  # Resized height for MediaPipe processing
DISABLE_HUD_EFFECTS = False  # Set True to turn off advanced overlays for benchmarking

# Hand Tracking Settings (MediaPipe)
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MAX_NUM_HANDS = 1         # Focus on a single primary hand for cursor/gestures

# Gesture Recognition Settings
PINCH_THRESHOLD = 0.25    # Ratio of thumb-to-index distance vs palm width (2D pixel space)
SWIPE_THRESHOLD_PX = 80   # Minimum pixel movement to qualify as a swipe
SWIPE_TIME_LIMIT_S = 0.4  # Time window to complete a swipe movement
SWIPE_COOLDOWN_S = 0.8    # Time to wait before triggering another swipe

# Virtual Mouse Settings
MOUSE_EMA_ALPHA = 0.12         # Baseline smoothing factor (lower = smoother, less jitter)
MOUSE_SPEED_MULTIPLIER = 1.0   # Scale factor for cursor motion
MOUSE_ZONE_BORDER_X = 0.10     # X border padding (10%) for edge stabilization
MOUSE_ZONE_BORDER_Y = 0.12     # Y border padding (12%)
MOUSE_ADAPTIVE_SMOOTHING = True # Enable dynamic alpha adjusting based on velocity
DOUBLE_CLICK_THRESHOLD_S = 0.35 # Double click time window in seconds

# Gesture Priority Hierarchy
GESTURE_PRIORITY = [
    "PINCH",
    "THUMBS_UP",
    "PEACE_SIGN",
    "POINT",
    "OPEN_PALM",
    "CLOSED_FIST"
]

# Hover Interactions
HOVER_ACTIVATION_TIME_S = 0.5  # Time in seconds to hover to activate click/mode change

# Styling & Theme (Futuristic Neon Cyberpunk Palette - BGR Format)
COLOR_BG = (15, 10, 5)          # Very dark grey-blue background for UI boxes
COLOR_BORDER = (242, 166, 0)    # Neon Cyan/Light Blue
COLOR_ACCENT = (147, 20, 255)   # Neon Purple/Magenta
COLOR_SUCCESS = (0, 242, 97)    # Neon Green
COLOR_WARNING = (0, 165, 255)   # Orange
COLOR_TEXT_PRIMARY = (255, 255, 255) # White
COLOR_TEXT_MUTED = (150, 150, 150)   # Light Grey
COLOR_SKELETON_JOINTS = (255, 0, 255) # Magenta for joint points
COLOR_SKELETON_LINES = (242, 166, 0)  # Cyan for bone lines

# Font settings
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_HUD = 0.5
FONT_THICKNESS = 1
