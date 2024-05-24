import os 
import time

from dotenv import load_dotenv
from openai import OpenAI
import pygame 

load_dotenv() 
last_played = {}


def play(object: str):
    print(f"Playing sound for: {object}")
    last_played[object] = time.time()


class OpenAITextToSpeech:

    _instance = None
    SECRET_KEY=os.getenv("SECRET_KEY")

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.client = OpenAI(api_key=self.SECRET_KEY)

    def generate_speech(self, text: str, path: str):
        with self.client.audio.speech.with_streaming_response.create(

        # output_path = os.path.join(os.getcwd(), "output", path)
        # response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        ) as response:
            response.stream_to_file(path)

        AudioPlayer(file=path).play()

class AudioPlayer:
    
    def __init__(self, file: str, ):
        self.file = file

    def play(self, frequency: int=44100):
        pygame.mixer.init(frequency=frequency)
        pygame.mixer.music.load(self.file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


# open_ai = OpenAITextToSpeech()
# open_ai.generate_speech(text="Hackaburg is awesome", path="output.mp3")
