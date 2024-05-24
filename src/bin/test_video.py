from pathlib import Path
import cv2
import asyncio


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
        # Read until the video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
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
