import time
from typing import List
from pydantic import BaseModel

from src.tts import AudioPlayer


class YoloLLMAssistant:

    # Initialize the assistant
    def __init__(self, audio_player: AudioPlayer):
        self.audio_player = audio_player

    # Keep track of predictions, and the number of predictions
    predictions: dict = {}

    def add_predictions(self, classes: List[str]):
        time_now_ms = time.time() * 1000  # Current time in milliseconds

        # Add the predictions to the predictions dictionary + 1
        for classname in classes:
            if classname not in self.predictions:
                self.predictions[classname] = {"ticks": 0, "last_seen": time_now_ms}

            self.predictions[classname]["ticks"] += 1
            self.predictions[classname]["last_seen"] = time_now_ms

        for classname in list(self.predictions.keys()):
            if classname not in classes:
                ticks = self.predictions[classname]["ticks"] - 1
                if ticks <= 0:
                    del self.predictions[classname]
                else:
                    self.predictions[classname]["ticks"] = ticks

    def print_predictions(self):
        print(self.predictions)
        return

    def analyze_predictions(
        self,
    ):
        pass
