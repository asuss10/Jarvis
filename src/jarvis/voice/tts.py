"""Text-to-speech with Piper.

The voice loads lazily on first use and can be unloaded to free memory.
Audio is synthesized to a temporary WAV file and played with winsound (Windows).
"""

import logging
import re
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 600

VoiceLoader = Callable[[Path], Any]
Player = Callable[[Path], None]


class TTSError(Exception):
    """Raised when speech cannot be produced."""


def clean_for_speech(text: str) -> str:
    """Strip Markdown and other things that sound wrong when read aloud."""
    text = re.sub(r"```.*?```", " code omitted ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # bullet markers
    text = re.sub(r"[*`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SPOKEN_CHARS:
        text = text[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0] + "..."
    return text


def _load_piper_voice(model_path: Path) -> Any:
    from piper import PiperVoice  # imported here: it is a heavy import

    return PiperVoice.load(str(model_path))


def _play_wav(wav_path: Path) -> None:
    import winsound  # Windows only; blocks until playback finishes

    winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)


class Speaker:
    def __init__(
        self,
        model_path: Path,
        voice_loader: VoiceLoader = _load_piper_voice,
        player: Player = _play_wav,
    ) -> None:
        self._model_path = model_path
        self._voice_loader = voice_loader
        self._player = player
        self._voice: Any = None

    def _get_voice(self) -> Any:
        if self._voice is None:
            if not self._model_path.exists():
                raise TTSError(f"Voice model not found: {self._model_path}")
            logger.info("Loading voice %s", self._model_path.name)
            try:
                self._voice = self._voice_loader(self._model_path)
            except Exception as exc:
                raise TTSError(f"Could not load voice: {exc}") from exc
        return self._voice

    def speak(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        voice = self._get_voice()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav_path = Path(tmp) / "speech.wav"
                with wave.open(str(wav_path), "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file)
                self._player(wav_path)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Speech failed: {exc}") from exc

    def unload(self) -> None:
        """Release the voice model from memory."""
        self._voice = None