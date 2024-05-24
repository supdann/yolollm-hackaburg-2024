from pathlib import Path
from queue import Queue
import threading
import time
from openai import BaseModel
from typing import List, Optional
import base64
import os
from pydantic import Field
import requests
import dotenv

from src.tts import AudioPlayer


# Langchain Imports
from langchain.output_parsers import PydanticOutputParser


# API key
api_key = os.getenv("OPENAI_API_KEY")


class GPTImageAnalysis(BaseModel):
    message: str = Field(
        description="The advice given by the assistant for the blind person"
    )


class YoloLLMAssistant:

    # Initialize the assistant
    def __init__(self, audio_player: AudioPlayer):
        self.audio_player = audio_player

        self.assistant_queue = Queue()  # Queue to manage audio playback requests
        self.worker_thread = threading.Thread(target=self._assistant_worker)
        self.worker_thread.daemon = True  # Allow thread to exit when main program exits
        self.worker_thread.start()

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

    def gen_and_play_instruction(
        self, classes: List[str], img_base64: Optional[str] = None
    ):
        self.assistant_queue.put((classes, img_base64))

    def _assistant_worker(self):
        while True:
            item = self.assistant_queue.get()
            classes, img_base64 = item

            if classes is None:
                break  # Exit if None is received
            self._gen_and_play_instruction(classes, img_base64)
            self.assistant_queue.task_done()

    def _gen_and_play_instruction(
        self, classes: List[str], img_base64: Optional[str] = None
    ):
        print(f"Generating instruction for {classes}")
        # Simulate the task taking 10 seconds
        description = self.describe(img_base64)
        self.audio_player._play_audio(description.message)

    def print_predictions(self):
        print(self.predictions)
        return

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe(self, img_b64: str) -> Optional[GPTImageAnalysis]:
        try:
            parser = PydanticOutputParser(pydantic_object=GPTImageAnalysis)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            format_instructions = parser.get_format_instructions()

            MAX_WORDS = 15

            describe_prompt = (
                # f"Describe in simple words and in less than f{MAX_WORDS} what is in the image. "
                f"Describe the image or warn about any potential danger or obstacles if necessary. "
                "For example, watch out for the bycicle coming on the left etc. "
                f"EVERYTHING IN LESS THAN {MAX_WORDS} WORDS. "
                "Format the sentence in such a way as if you were speaking directly to a blind person. "
            )

            # Add Format Instructions to the prompt
            describe_prompt += f"\n\n{format_instructions}"

            payload = {
                "model": "gpt-4o",
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are an AI assistant for a blind person. "
                                    "You received an image and you need to describe it and warn about potential danger or obstacles."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": describe_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 300,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            response = response.json()["choices"][0]["message"]["content"]

            data = parser.parse(response)

            print(data)

            return data

        except Exception as e:
            print(e)
            return None

    def analyze_predictions(
        self,
        with_img: Optional[Path] = None,
    ):
        if len(self.predictions) == 0:
            return

        img_b64 = None

        if with_img:
            img_b64 = self.encode_image(with_img)
            print(f"Image: {len(img_b64)}")

        classes_str = list(self.predictions.keys())
        self.gen_and_play_instruction(classes_str, img_b64)
