import wave
from pathlib import Path

import pytest

from jarvis.voice.tts import MAX_SPOKEN_CHARS, Speaker, TTSError, clean_for_speech


class FakeVoice:
    def __init__(self):
        self.spoken = []

    def synthesize_wav(self, text, wav_file):
        self.spoken.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 100)


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "voice.onnx"
    path.write_bytes(b"fake")
    return path


def make_speaker(model_file, voice=None, played=None):
    voice = voice or FakeVoice()
    played = played if played is not None else []

    def player(wav_path: Path) -> None:
        with wave.open(str(wav_path), "rb") as wav:  # file must exist and be valid WAV
            played.append(wav.getnframes())

    loads = []

    def loader(path):
        loads.append(path)
        return voice

    return Speaker(model_file, voice_loader=loader, player=player), voice, played, loads


def test_markdown_is_stripped():
    assert clean_for_speech("**Hello** `world`") == "Hello world"


def test_bullets_and_headers_removed():
    assert clean_for_speech("# Title\n- one\n- two") == "Title one two"


def test_code_blocks_and_urls_are_replaced():
    result = clean_for_speech("See https://example.com and ```print(1)``` ok")
    assert "http" not in result and "print" not in result


def test_long_text_is_truncated():
    result = clean_for_speech("word " * 500)
    assert len(result) <= MAX_SPOKEN_CHARS + 3
    assert result.endswith("...")


def test_voice_loads_lazily(model_file):
    speaker, _, _, loads = make_speaker(model_file)
    assert loads == []
    speaker.speak("hello")
    speaker.speak("again")
    assert len(loads) == 1


def test_speak_synthesizes_and_plays_cleaned_text(model_file):
    speaker, voice, played, _ = make_speaker(model_file)
    speaker.speak("**Hi** there")
    assert voice.spoken == ["Hi there"]
    assert played == [100]


def test_empty_text_is_not_spoken(model_file):
    speaker, voice, played, loads = make_speaker(model_file)
    speaker.speak("  ** ")
    assert voice.spoken == [] and played == [] and loads == []


def test_unload_forces_reload(model_file):
    speaker, _, _, loads = make_speaker(model_file)
    speaker.speak("one")
    speaker.unload()
    speaker.speak("two")
    assert len(loads) == 2


def test_missing_model_raises_clear_error(tmp_path):
    speaker = Speaker(tmp_path / "missing.onnx", voice_loader=lambda p: None, player=lambda p: None)
    with pytest.raises(TTSError, match="not found"):
        speaker.speak("hello")


def test_synthesis_failure_becomes_tts_error(model_file):
    class BrokenVoice:
        def synthesize_wav(self, text, wav_file):
            raise RuntimeError("boom")

    speaker, _, _, _ = make_speaker(model_file, voice=BrokenVoice())
    with pytest.raises(TTSError):
        speaker.speak("hello")