from jarvis.voice.tts import ensure_voice


def test_existing_voice_is_not_downloaded(tmp_path):
    model = tmp_path / "my-voice.onnx"
    model.write_bytes(b"fake")
    assert ensure_voice("my-voice", tmp_path) == model