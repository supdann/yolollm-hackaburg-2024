# YOLOLLM

This project captures video input from the camera and processes frames using YOLO (You Only Look Once) with OpenCV.

## Prerequisites

- Python 3.x
- OpenCV
- Numpy
- YOLO configuration, weights, and class names files

## Installation

1. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Upgrade pip and setuptools:**

    ```bash
    pip install --upgrade pip setuptools
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download YOLO files:**

    Download the following files and place them in the models directory:
    - `yolov8n.pt`

## Usage

1. **Run the script:**

    Ensure your virtual environment is activated, and then run the script:

    ```bash
    python main.py
    ```

## Notes

- Ensure you have the necessary permissions to access the camera.
- The script is designed to process as many frames as possible in real-time, but performance may vary depending on your hardware.

## References

- [YOLO Website](https://pjreddie.com/darknet/yolo/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [COCO Dataset](https://cocodataset.org/)