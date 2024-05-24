import base64
import hashlib
import os
import threading
from typing import Optional
from pathlib import Path
from queue import Queue


import dotenv
from openai import BaseModel, OpenAI
from pydantic import Field
import pygame
import requests


# Langchain Imports
from langchain.output_parsers import PydanticOutputParser


dotenv.load_dotenv()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# API key
api_key = os.getenv("OPENAI_API_KEY")


class GPTImageAnalysis(BaseModel):
    message: str = Field(
        description="The advice given by the assistant for the blind person"
    )


def describe(image_path: str) -> Optional[GPTImageAnalysis]:

    try:

        parser = PydanticOutputParser(pydantic_object=GPTImageAnalysis)

        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        format_instructions = parser.get_format_instructions()

        describe_prompt = (
            "Describe in simple words and a short sentence what is in the image. "
            "Format the sentence in such a way as if you were speaking directly to a blind person. "
            f"\n\n{format_instructions}"
        )

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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        response = response.json()["choices"][0]["message"]["content"]

        data = parser.parse(response)

        print(data)

        return data

    except Exception as e:
        print(e)
        return None


class AudioPlayer:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self, frequency: int = 44100):
        pygame.mixer.init(frequency=frequency)
        self.audio_queue = Queue()  # Queue to manage audio playback requests
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True  # Allow thread to exit when main program exits
        self.audio_thread.start()

    def encode_image(self, image_path):
        # Function to encode the image
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def describe(self, image_path: str):
        # Getting the base64 string
        base64_image = self.encode_image(image_path)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.OPENAI_API_KEY}"}

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe in a very short and simple sentence what is in the image, for a person who is blind and cannot see.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        print(response.json())

        return response.json()["choices"][0]["message"]["content"]


    def gen_openai_audio(
        self, text: str, voice: str, audio_save_path: Path, file_name: str, format="mp3"
    ) -> Path:
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        speech_filepath = audio_save_path / f"{file_name}.{format}"
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            response_format=format,
            input=text,
        )
        response.stream_to_file(speech_filepath)
        return speech_filepath


    def hash_text(self, text: str):
        # Convert text to bytes
        text_bytes = text.encode('utf-8')
        
        # Choose a hashing algorithm (e.g., MD5, SHA-1, SHA-256)
        hash_algorithm = hashlib.sha256()
        
        # Update the hash object with the bytes of the text
        hash_algorithm.update(text_bytes)
        
        # Get the hexadecimal representation of the hash
        hashed_text = hash_algorithm.hexdigest()
        
        return hashed_text


    def get_file_for_description(self, description: str):
        # Audio Save Path
        audio_save_path = Path("data/description")
        
        # Create data/audio directory if it doesn't exist
        if not os.path.exists(audio_save_path):
            os.makedirs(audio_save_path)

        file_name = self.hash_text(description)
        file_path = audio_save_path / f"{file_name}.mp3"

        self.gen_openai_audio(
            text=f"{description}",
            voice="alloy",
            audio_save_path=audio_save_path,
            file_name=file_name,
        )

        # Return the path to the audio file
        return file_path

    def play(self, image_path: str):
        self.audio_queue.put(image_path)  # Add playback request to the queue

    def stop(self):
        self.audio_queue.put(None)  # Add None to the queue to stop the audio thread

    def _audio_worker(self):
        while True:
            image_path = self.audio_queue.get()  # Get next audio playback request
            if image_path is None:
                break  # Exit if None is received

            self._play_audio(image_path)
            self.audio_queue.task_done()

    def _play_audio(self, image_path: str):
        description = self.describe(image_path)
        filename = self.get_file_for_description(description)

        """
        Plays the specified audio file using pygame.mixer.music.

        Args:
            file (str): The path to the audio file to be played.
        """
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
  


# ap = AudioPlayer()
# ap.play("pave.jpg")
