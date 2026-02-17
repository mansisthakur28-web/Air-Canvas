"""
Air Canvas — Draw in the air using hand gestures!

A Computer Vision application that uses OpenCV and MediaPipe Hands to let
you draw on a virtual canvas by tracking your index finger. The drawing
is overlaid onto the live webcam feed in real time.

Controls:
  - Raise ONLY the index finger  → Drawing Mode (draw on canvas)
  - Raise index + middle fingers → Selection / Hover Mode (pick colors, buttons)
  - Press 'q' to quit

Author : Antigravity AI
Python : >=3.10
"""

import sys
import cv2
import numpy as np
import mediapipe as mp

try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except AttributeError:
    # Fallback for newer MediaPipe versions where 'solutions' isn't auto-imported
    import mediapipe.python.solutions.hands as mp_hands_module
    import mediapipe.python.solutions.drawing_utils as mp_draw_module
    mp_hands = mp_hands_module
    mp_draw = mp_draw_module
# Constants
# ──────────────────────────────────────────────

# Window and header dimensions
WINDOW_NAME = "Air Canvas"
HEADER_HEIGHT = 80              # Height of the top toolbar in pixels

# Default canvas size (will be updated to match camera resolution)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Drawing settings
DRAW_THICKNESS = 7              # Brush stroke thickness
ERASER_THICKNESS = 40           # Eraser stroke thickness (wider for easy erasing)

# Colour palette (BGR format for OpenCV)
# Modified to pure/bright colors for maximum clarity as requested
COLORS = {
    "BLUE":  (255, 0, 0),    # Pure Blue
    "GREEN": (0, 255, 0),    # Pure Green
    "RED":   (0, 0, 255),    # Pure Red
}
ERASER_COLOR = (0, 0, 0)       # Black — same as canvas background
PALM_ERASER_RADIUS = 60        # Radius for palm eraser tool

# Overlay blending weight (alpha for drawing layer)
# Increased to 1.0 (fully opaque) for maximum visibility
OVERLAY_ALPHA = 1.0

# ──────────────────────────────────────────────
# Button layout definition
# ──────────────────────────────────────────────

def _build_buttons(frame_width: int) -> list[dict]:
    """
    Build a list of header button definitions evenly distributed across
    the top of the frame.

    Each button is a dict with keys:
      - name  : display label
      - color : BGR fill colour for the button
      - x1, y1, x2, y2 : bounding-box coordinates

    Args:
        frame_width: Width of the video frame in pixels.

    Returns:
        A list of button definition dicts.
    """
    labels = ["CLEAR ALL", "BLUE", "GREEN", "RED", "ERASER"]
    btn_count = len(labels)
    btn_width = frame_width // btn_count
    padding = 5  # small gap between buttons

    buttons: list[dict] = []
    for i, label in enumerate(labels):
        x1 = i * btn_width + padding
        x2 = (i + 1) * btn_width - padding
        y1 = padding
        y2 = HEADER_HEIGHT - padding

        # Choose a fill colour that matches the label
        if label in COLORS:
            fill = COLORS[label]
        elif label == "ERASER":
            fill = (100, 100, 100)   # grey
        else:
            fill = (50, 50, 50)      # dark grey for CLEAR ALL

        buttons.append({
            "name": label,
            "color": fill,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
        })
    return buttons


# ──────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────

def initialize_camera(cam_index: int = 0) -> cv2.VideoCapture:
    """
    Open the default webcam and configure its resolution.

    Raises SystemExit if the camera cannot be opened.

    Args:
        cam_index: Index of the camera device (default 0).

    Returns:
        An opened cv2.VideoCapture object.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please check your camera connection.")
        sys.exit(1)

    # Request a reasonable resolution (camera may choose the closest supported one)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    return cap


# ──────────────────────────────────────────────
# Header / UI drawing
# ──────────────────────────────────────────────

def create_header(frame: np.ndarray, buttons: list[dict], active_label: str) -> None:
    """
    Draw the persistent toolbar at the top of *frame* (in-place).

    The currently selected button is highlighted with a white border.

    Args:
        frame:        The video frame to draw on.
        buttons:      List of button definitions (from _build_buttons).
        active_label: The label of the currently active button/tool.
    """
    # Semi-transparent dark overlay behind the header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], HEADER_HEIGHT), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for btn in buttons:
        # Draw filled rectangle for each button
        cv2.rectangle(frame, (btn["x1"], btn["y1"]),
                      (btn["x2"], btn["y2"]), btn["color"], -1)

        # Highlight the active button with a white border
        if btn["name"] == active_label:
            cv2.rectangle(frame, (btn["x1"], btn["y1"]),
                          (btn["x2"], btn["y2"]), (255, 255, 255), 3)

        # Centre the text label inside the button
        text = btn["name"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = btn["x1"] + (btn["x2"] - btn["x1"] - tw) // 2
        ty = btn["y1"] + (btn["y2"] - btn["y1"] + th) // 2
        cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness)


# ──────────────────────────────────────────────
# Hand / finger detection
# ──────────────────────────────────────────────

def detect_fingers(hand_landmarks) -> list[bool]:
    """
    Determine which fingers are raised based on MediaPipe hand landmarks.

    The method compares the y-coordinate of each fingertip to the
    corresponding PIP (proximal interphalangeal) joint. Because the
    y-axis is inverted in image coordinates, a fingertip with a *smaller*
    y value than its PIP joint means the finger is raised.

    For the thumb, the x-coordinate comparison is used instead (tip vs.
    IP joint), taking hand orientation into account.

    Args:
        hand_landmarks: A single hand's NormalizedLandmarkList from
                        MediaPipe Hands.

    Returns:
        A list of 5 booleans [thumb, index, middle, ring, pinky]:
        True if the finger is raised, False otherwise.
    """
    landmarks = hand_landmarks.landmark

    # Landmark indices — tip and the joint just below it
    tip_ids = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
    pip_ids = [3, 6, 10, 14, 18]   # corresponding comparison joints

    fingers: list[bool] = []

    for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
        if i == 0:
            # Thumb: compare x-coordinates.
            # If wrist x < middle-finger MCP x the hand is "normal" (palm facing camera)
            wrist_x = landmarks[0].x
            middle_mcp_x = landmarks[9].x
            if wrist_x < middle_mcp_x:
                # Right hand (or left hand palm-away): thumb open when tip.x > pip.x
                fingers.append(landmarks[tip].x > landmarks[pip].x)
            else:
                fingers.append(landmarks[tip].x < landmarks[pip].x)
        else:
            # Other fingers: compare y-coordinates (lower y = higher on screen)
            fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers


def get_drawing_mode(fingers: list[bool]) -> str:
    """
    Decide the current interaction mode from the finger state.

    Modes:
      "erase_palm" — All 5 fingers are raised.
      "select"     — Index and middle fingers are raised.
      "draw"       — Only the index finger is raised.
      "idle"       — Any other configuration.

    Args:
        fingers: List of 5 booleans from detect_fingers().

    Returns:
        One of "erase_palm", "select", "draw", or "idle".
    """
    # If all fingers are raised, activate palm eraser
    if all(fingers):
        return "erase_palm"

    index_up = fingers[1]
    middle_up = fingers[2]

    if index_up and middle_up:
        return "select"
    elif index_up:
        return "draw"
    else:
        return "idle"


# ──────────────────────────────────────────────
# Selection / button handling
# ──────────────────────────────────────────────

def handle_selection(x: int, y: int, buttons: list[dict],
                     current_color: tuple, eraser_on: bool
                     ) -> tuple[tuple, bool, bool, str]:
    """
    Check whether the fingertip position (x, y) falls on a header button
    and return the updated drawing state.

    Args:
        x, y:          Fingertip pixel coordinates.
        buttons:       List of button dicts.
        current_color: Current drawing colour (BGR tuple).
        eraser_on:     Whether eraser mode is active.

    Returns:
        A 4-tuple of (new_color, eraser_on, should_clear, active_label).
    """
    should_clear = False
    active_label = ""

    # Determine current label based on state
    if eraser_on:
        active_label = "ERASER"
    else:
        for name, bgr in COLORS.items():
            if bgr == current_color:
                active_label = name
                break

    # Only respond to clicks inside the header region
    if y > HEADER_HEIGHT:
        return current_color, eraser_on, should_clear, active_label

    for btn in buttons:
        if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
            name = btn["name"]
            if name == "CLEAR ALL":
                should_clear = True
                # Keep the same colour / eraser state
            elif name == "ERASER":
                eraser_on = True
                active_label = "ERASER"
            elif name in COLORS:
                current_color = COLORS[name]
                eraser_on = False
                active_label = name
            break

    return current_color, eraser_on, should_clear, active_label


# ──────────────────────────────────────────────
# Canvas drawing
# ──────────────────────────────────────────────

def draw_on_canvas(canvas: np.ndarray,
                   prev_point: tuple[int, int] | None,
                   curr_point: tuple[int, int],
                   color: tuple,
                   eraser: bool) -> tuple[int, int]:
    """
    Draw a line segment from *prev_point* to *curr_point* on the canvas.

    If eraser mode is active the line is drawn in ERASER_COLOR with a
    thicker stroke so it effectively "erases" existing content.

    Args:
        canvas:     The transparent drawing layer (NumPy array).
        prev_point: Previous fingertip position (x, y), or None if this
                    is the first point of a new stroke.
        curr_point: Current fingertip position (x, y).
        color:      BGR colour tuple to draw with.
        eraser:     True to use eraser mode.

    Returns:
        The current point (to be stored as prev_point for the next frame).
    """
    draw_color = ERASER_COLOR if eraser else color
    thickness = ERASER_THICKNESS if eraser else DRAW_THICKNESS

    if prev_point is not None:
        cv2.line(canvas, prev_point, curr_point, draw_color, thickness)
    return curr_point


# ──────────────────────────────────────────────
# Canvas overlay
# ──────────────────────────────────────────────

def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    Blend the transparent drawing canvas onto the live webcam frame.

    Only the non-black pixels of the canvas are blended so that the
    background remains clear.

    For the eraser (black strokes), we directly mask those pixels out
    of the frame so erased areas show the raw webcam feed.

    Args:
        frame:  The current webcam frame (BGR).
        canvas: The drawing layer (BGR, black background).

    Returns:
        A new frame with the canvas drawings composited on top.
    """
    # Create a mask of all non-black pixels on the canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Isolate the region of the frame where drawing exists and blend
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    # Weighted blend for a translucent drawing effect
    combined = cv2.addWeighted(frame_bg, 1.0, canvas_fg, OVERLAY_ALPHA, 0)

    # Where there was no drawing, keep the original frame pixels
    result = cv2.add(
        cv2.bitwise_and(frame, frame, mask=mask_inv),
        cv2.bitwise_and(combined, combined, mask=mask)
    )

    # Simpler but effective: blend the entire canvas on top
    result = cv2.addWeighted(frame, 1 - OVERLAY_ALPHA, canvas, OVERLAY_ALPHA, 0)

    # Re-paste original frame where canvas is black (transparent)
    result = np.where(canvas > 0, result, frame)

    return result


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def main() -> None:
    """
    Entry point.  Sets up camera, MediaPipe Hands, canvas, and runs
    the main event loop that reads frames, detects hand gestures, and
    draws / selects accordingly.
    """
    # ── Initialise camera ──
    cap = initialize_camera()
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read initial frame from camera.")
        cap.release()
        sys.exit(1)

    frame_h, frame_w = test_frame.shape[:2]
    print(f"[INFO] Camera opened — resolution {frame_w}×{frame_h}")

    # ── Initialise MediaPipe Hands ──
    # Note: mp_hands is already imported/resolved at the top of the file
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    # mp_draw is also resolved at the top

    # ── Create transparent drawing canvas (all black = transparent) ──
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # ── Build header buttons ──
    buttons = _build_buttons(frame_w)

    # ── State variables ──
    draw_color: tuple = COLORS["BLUE"]   # default colour
    eraser_on: bool = False
    active_label: str = "BLUE"
    prev_point: tuple[int, int] | None = None

    print("[INFO] Air Canvas is running. Press 'q' to quit.")

    while True:
        # ── Read frame ──
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame — camera may have disconnected.")
            break

        # Flip horizontally for a mirror-like experience
        frame = cv2.flip(frame, 1)

        # ── Hand detection ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw hand skeleton on the frame (visual feedback)
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                # Get index-finger tip coordinates (landmark 8)
                ix = int(hand_lms.landmark[8].x * frame_w)
                iy = int(hand_lms.landmark[8].y * frame_h)

                # Determine which fingers are raised
                fingers = detect_fingers(hand_lms)
                mode = get_drawing_mode(fingers)

                if mode == "erase_palm":
                    # ── Palm Eraser Mode ──
                    prev_point = None

                    # Use the middle finger MCP (landmark 9) as the palm center
                    px = int(hand_lms.landmark[9].x * frame_w)
                    py = int(hand_lms.landmark[9].y * frame_h)

                    # Erase on canvas (draw large black circle)
                    cv2.circle(canvas, (px, py), PALM_ERASER_RADIUS, ERASER_COLOR, -1)

                    # Visual feedback on frame: outline required to see the eraser
                    cv2.circle(frame, (px, py), PALM_ERASER_RADIUS, (255, 255, 255), 2)
                    cv2.putText(frame, "ERASE", (px - 20, py - PALM_ERASER_RADIUS - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                elif mode == "select":
                    # ── Selection / hover mode ──
                    prev_point = None  # reset so strokes don't connect

                    draw_color, eraser_on, should_clear, active_label = \
                        handle_selection(ix, iy, buttons, draw_color, eraser_on)

                    if should_clear:
                        canvas[:] = 0  # wipe the entire canvas

                    # Visual indicator: circle at fingertip
                    cv2.circle(frame, (ix, iy), 15, (255, 255, 0), 2)

                elif mode == "draw":
                    # ── Drawing mode ──
                    # Do not draw inside the header area
                    if iy > HEADER_HEIGHT:
                        prev_point = draw_on_canvas(
                            canvas, prev_point, (ix, iy),
                            draw_color, eraser_on
                        )
                    else:
                        prev_point = None

                    # Visual indicator: filled circle at fingertip
                    indicator_color = ERASER_COLOR if eraser_on else draw_color
                    cv2.circle(frame, (ix, iy), 8, indicator_color, -1)

                else:
                    # ── Idle ── (no relevant fingers raised)
                    prev_point = None
        else:
            # No hand detected — reset previous point
            prev_point = None

        # ── Composite the canvas onto the frame ──
        frame = overlay_canvas(frame, canvas)

        # ── Draw the header on top (always visible) ──
        create_header(frame, buttons, active_label)

        # ── Display ──
        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting Air Canvas.")
            break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] Resources released. Goodbye!")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
