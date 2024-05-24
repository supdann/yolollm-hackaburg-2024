import os
import threading
from queue import Queue
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pygame

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AudioPlayer:

    def __init__(self, frequency: int = 44100):
        pygame.mixer.init(frequency=frequency)
        self.audio_queue = Queue()  # Queue to manage audio playback requests
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True  # Allow thread to exit when main program exits
        self.audio_thread.start()

    def gen_openai_audio(
        self, text: str, voice: str, audio_save_path: Path, file_name: str, format="mp3"
    ) -> Path:
        client = OpenAI(api_key=OPENAI_API_KEY)
        speech_filepath = audio_save_path / f"{file_name}.{format}"
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            response_format=format,
            input=text,
        )
        response.stream_to_file(speech_filepath)
        return speech_filepath

    def get_file_for_class(self, class_name: str):

        # Audio Save Path
        audio_save_path = Path("./src/data/audio")

        # Create data/audio directory if it doesn't exist
        if not os.path.exists(audio_save_path):
            os.makedirs(audio_save_path)

        file_path = audio_save_path / f"{class_name}.mp3"

        if not file_path.exists():
            self.gen_openai_audio(
                text=f"{class_name}",
                voice="alloy",
                audio_save_path=audio_save_path,
                file_name=class_name,
            )

        # Return the path to the audio file
        return file_path

    def play(self, classname: str):
        self.audio_queue.put(classname)  # Add playback request to the queue

    def stop(self):
        self.audio_queue.put(None)  # Add None to the queue to stop the audio thread

    def _audio_worker(self):
        while True:
            classname = self.audio_queue.get()  # Get next audio playback request
            if classname is None:
                break  # Exit if None is received

            self._play_audio(classname)
            self.audio_queue.task_done()

    def _play_audio(self, classname: str):
        filename = self.get_file_for_class(classname)

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
