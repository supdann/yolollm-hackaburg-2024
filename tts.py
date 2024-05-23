import time

last_played = {}

def play(object: str):
    print(f"Playing sound for: {object}")
    last_played[object] = time.time()