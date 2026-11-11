"""Speak a sentence with Piper.

Usage:
    python scripts/say.py "Hello there"
    python scripts/say.py --voice en_GB-alan-medium
"""

import argparse
import sys
import time

from jarvis.config import PROJECT_ROOT
from jarvis.logging_config import setup_logging
from jarvis.voice.tts import Speaker, TTSError

VOICES_DIR = PROJECT_ROOT / "data" / "voices"
DEFAULT_VOICE = "en_US-lessac-medium"
DEFAULT_TEXT = "Good evening. All systems are running normally. How can I help you today?"


def ensure_voice(voice_id: str):
    model_path = VOICES_DIR / f"{voice_id}.onnx"
    if not model_path.exists():
        from piper.download_voices import download_voice

        print(f"Downloading voice {voice_id} ...")
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        download_voice(voice_id, VOICES_DIR)
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    args = parser.parse_args()

    setup_logging("INFO")
    try:
        model_path = ensure_voice(args.voice)
    except Exception as exc:
        print(f"[error] Could not get voice '{args.voice}': {exc}")
        return 1

    timings: dict[str, float] = {}

    def timed_player(wav_path):
        import winsound

        timings["synthesis_done"] = time.perf_counter()
        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)

    speaker = Speaker(model_path, player=timed_player)
    try:
        started = time.perf_counter()
        speaker._get_voice()
        print(f"Voice loaded in {time.perf_counter() - started:.2f}s")

        started = time.perf_counter()
        speaker.speak(args.text)
        finished = time.perf_counter()
        synthesis = timings["synthesis_done"] - started
        print(f"Synthesis: {synthesis:.2f}s | Playback: {finished - timings['synthesis_done']:.2f}s")
    except TTSError as exc:
        print(f"[error] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())