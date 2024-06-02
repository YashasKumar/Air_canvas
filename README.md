# Air Canvas with Hand Tracking

This project implements an air canvas using Mediapipe and OpenCV. Users can draw on a virtual canvas by moving their index finger in front of a webcam. The project includes a feature to start a new drawing segment when the thumb is close to the index finger, creating breaks in the drawing sequence.

## Features

- **Real-time Hand Tracking**: Uses Mediapipe to track hand landmarks.
- **Drawing on Virtual Canvas**: Draw on a virtual canvas using your index finger.
- **Segmented Drawing**: Automatically create new drawing segments when the thumb is close to the index finger.
- **Mode Switching**: Switch between drawing mode and viewing mode.
- **Save Drawings**: Save the current canvas as an image file.
- **Clear Canvas**: Clear the canvas with a key press.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- Numpy

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/YashasKumar/Air_canvas.git
   cd Air-canvas
   ```

2. **Install dependencies**:
   ```sh
   pip install opencv-python mediapipe numpy
   ```

## Usage

Run the script:
```sh
python aircanvas.py
```

### Key Controls

- **'d'**: Enable drawing mode.
- **'r'**: Enable viewing mode.
- **'c'**: Clear the canvas.
- **'q'**: Save the canvas and quit.

## License

This project is licensed under the MIT License.
