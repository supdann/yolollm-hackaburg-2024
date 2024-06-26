from pathlib import Path
import cv2
import asyncio

import dotenv
import math
import sys
import os
import time
import logging
from ultralytics import YOLO

from src.assistant import YoloLLMAssistant
from src.tts import AudioPlayer

# Load environment variables
dotenv.load_dotenv()

print(cv2.__version__)

# Set logging level to ERROR to suppress YOLO informational messages
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load a model
model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)

# List of objects from the YOLO model that can trigger danger warnings
trigger_objects = ["bottle", "bicycle", "car"]

last_played = {}

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


# Video processing interval in milliseconds
PROCESS_INTERVAL_MS = 600

# Delay between sound playback
PLAY_DELAY_SECONDS = 5


async def start():

    # Path to the .mov video file
    video_path = Path("./src/data/video") / "test_video.mov"

    # Convert Path object to string
    video_path_str = str(video_path)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path_str)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        audioplayer = AudioPlayer()
        assistant = YoloLLMAssistant(audio_player=audioplayer)
        last_process_time = 0  # Track the last process time

        # Read until the video is completed
        while cap.isOpened():
            current_time = time.time() * 1000  # Current time in milliseconds
            # Capture frame-by-frame
            success, frame = cap.read()
            if success:

                if current_time - last_process_time >= PROCESS_INTERVAL_MS:
                    last_process_time = current_time

                    with SuppressOutput():
                        # Run batched inference on the frame
                        results = model(frame, stream=True)

                    # Placeholder for predicted classes
                    predicted_classes = []

                    # Process results list
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:

                            # Class name
                            cls = int(box.cls[0])
                            class_name = classNames[cls]

                            # Check if the class name is in trigger objects
                            if class_name in trigger_objects:
                                # Bounding box
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = (
                                    int(x1),
                                    int(y1),
                                    int(x2),
                                    int(y2),
                                )  # convert to int values

                                # Put box in frame
                                cv2.rectangle(
                                    frame, (x1, y1), (x2, y2), (255, 0, 255), 3
                                )

                                # Confidence
                                # confidence = math.ceil((box.conf[0] * 100)) / 100

                                predicted_classes.append(class_name)
                                # current_time = time.time()
                                # if (
                                #     class_name not in last_played
                                #     or current_time - last_played[class_name]
                                #     > PLAY_DELAY_SECONDS
                                # ):
                                #     print(f"attempting to play {class_name} ")
                                #     audioplayer.play(class_name)
                                #     last_played[class_name] = current_time

                                # Object details
                                org = [x1, y1]
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                fontScale = 1
                                color = (255, 0, 0)
                                thickness = 2

                                cv2.putText(
                                    frame,
                                    classNames[cls],
                                    org,
                                    font,
                                    fontScale,
                                    color,
                                    thickness,
                                )

                    # If the classes are trigger objects, add them to the predictions
                    assistant.add_predictions(predicted_classes)

                    # Print the predictions
                    assistant.print_predictions()

                    # Save the current frame as an image
                    image_path = Path("./src/data") / f"captured_frame.jpg"
                    cv2.imwrite(str(image_path), frame)

                    # Analyze the predictions
                    assistant.analyze_predictions(with_img=image_path)

                # Display the resulting frame
                cv2.imshow("Video Playback", frame)

                # Press 'q' on the keyboard to exit the playback
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release the video capture object
        cap.release()

        # Close all the frames
        cv2.destroyAllWindows()


def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
