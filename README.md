# Dev Guide

## Usage for VSCode

For everyone who uses VSCode, this should work almost out of the box. Just open the project folder in VSCode, make sure you have Docker installed,
and you should be able to "Open in Container" from the Remote Development extension. (You may need to install the extension first.)

## Usage for Docker

If you don't use VSCode, you can still use Docker to run the project. Just run the following command in the project directory:

```bash
docker build -t yolollm .devcontainer/Dockerfile
docker run -it --rm -v $(pwd):/app yolollm
```

This will build the Docker image and run a container with the project mounted at `/app`. You can then run the project as usual. (Connect to the container, activate the virtual environment, and run the script.)

**Note:** Remember to install the required dependencies in the virtual environment by running `poetry install --no-root` before running the script.


## Adding New Dependencies

If you need to add a new dependency, you can do so by running the following command:

```bash
poetry add <package>
```

This will add the package to the `pyproject.toml` file and install it in the virtual environment.



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