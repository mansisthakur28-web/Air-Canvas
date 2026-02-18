# Air Canvas

**Draw in the air using hand gestures!**

Air Canvas is a Computer Vision application that uses OpenCV and MediaPipe Hands to let you draw on a virtual canvas by tracking your index finger. The drawing is overlaid onto the live webcam feed in real time, creating an augmented reality drawing experience.

## Features

- **Draw in the Air**: Use your index finger as a virtual pen.
- **Color Selection**: Choose from Blue, Green, and Red colors using on-screen buttons.
- **Eraser Tool**: Erase mistakes with a dedicated eraser tool.
- **Clear All**: Quickly clear the entire canvas.
- **Palm Eraser**: Use your open hand (all fingers up) to wipe large areas.
- **Real-time Interaction**: Smooth tracking and rendering.

## Requirements

- Python >= 3.10
- Webcam

## Installation

1.  Clone the repository or download the source code.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the application:

    ```bash
    python air_canvas.py
    ```

2.  **Controls**:

    -   **Drawing Mode**: Raise **ONLY** your index finger to draw.
    -   **Selection / Hover Mode**: Raise **Index + Middle** fingers to move the cursor without drawing (use this to select colors or buttons).
    -   **Palm Eraser**: Raise **ALL** fingers to use the large palm eraser.
    -   **Quit**: Press `q` to exit the application.

## How it Works

The application uses Google's MediaPipe framework for hand tracking. It detects hand landmarks to identify gestures:
-   **Index finger up**: Creative mode (Drawing).
-   **Index + Middle finger up**: Navigation mode (Selection).
-   **Open hand**: Erasing mode.

The drawing is performed on a separate transparent canvas layer and then merged with the webcam feed using OpenCV.

## Dependencies

-   `opencv-python`
-   `mediapipe`
-   `numpy`

See `requirements.txt` for specific versions.
