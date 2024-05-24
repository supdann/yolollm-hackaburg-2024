import dotenv
import cv2
import math
import sys
import os
import time
import logging
import asyncio
from ultralytics import YOLO
from fastapi import FastAPI
import uvicorn
from tts import play, last_played
from gpt import describe

# Load environment variables
dotenv.load_dotenv()

print(cv2.__version__)

# Set logging level to ERROR to suppress YOLO informational messages
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load a model
model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)

# List of objects that can trigger the playing
trigger_objects = ["bottle"]

# classes
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Initialize FastAPI app
app = FastAPI()


# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


@app.get("/describe")
def get_description():
    success, img = cap.read()
    if success:
        image_path = "data/captured_frame.jpg"
        cv2.imwrite(image_path, img)
        return {"description": describe(image_path)}
    else:
        return {"error": "Failed to capture image"}


# Function to run the FastAPI app
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Function to run the camera processing loop
async def run_camera():

    while True:
        for i in range(2):
            success, img = cap.read()

        with SuppressOutput():
            # Run batched inference on a list of images
            results = model(img, stream=True)

        # Process results list
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )  # convert to int values

                # Put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Check if the class name is in trigger objects
                if class_name in trigger_objects:
                    current_time = time.time()
                    if (
                        class_name not in last_played
                        or current_time - last_played[class_name] > 10
                    ):
                        play(class_name)

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(
                    img, classNames[cls], org, font, fontScale, color, thickness
                )

        cv2.imshow("Webcam", img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Yield control to the event loop
        await asyncio.sleep(0.01)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Main function to run both tasks concurrently
async def main():
    server_task = asyncio.create_task(asyncio.to_thread(start_server))
    camera_task = asyncio.create_task(run_camera())

    await asyncio.gather(server_task, camera_task)


if __name__ == "__main__":
    asyncio.run(main())
