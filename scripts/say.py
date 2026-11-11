"""Speak a sentence with Piper. Usage: python scripts/say.py "Hello there" """

import argparse
import sys
import time

from jarvis.config import PROJECT_ROOT
from jarvis.logging_config import setup_logging
from jarvis.voice.tts import Speaker, TTSError

DEFAULT_MODEL = PROJECT_ROOT / "data" / "voices" / "en_US-lessac-medium.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="Hello. I am Jarvis, and this is my voice.")
    args = parser.parse_args()

    setup_logging("INFO")
    speaker = Speaker(DEFAULT_MODEL)
    try:
        started = time.perf_counter()
        speaker._get_voice()
        print(f"Voice loaded in {time.perf_counter() - started:.2f}s")

        started = time.perf_counter()
        speaker.speak(args.text)
        print(f"Synthesis + playback took {time.perf_counter() - started:.2f}s")
    except TTSError as exc:
        print(f"[error] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())