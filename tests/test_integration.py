"""Integration tests for speaker recognition API and client."""

import base64
import wave
from multiprocessing import Process
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn

from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.api import app
from speaker_recognition.models import (
    AudioInput,
    RecognitionRequest,
    TrainingRequest,
    VoiceSample,
)
from speaker_recognition.recognizer import recognizer

EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_data"
API_HOST = "127.0.0.1"
API_PORT = 8765
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


def start_api_server():
    """Start the API server in a subprocess."""
    from speaker_recognition.api import app

    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="error")


def read_audio_file_as_base64(file_path: Path) -> tuple[str, int]:
    """Read WAV file and encode PCM data as base64.

    Args:
        file_path: Path to the WAV audio file

    Returns:
        Tuple of (Base64 encoded PCM audio data, actual sample rate)
    """
    with wave.open(str(file_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm_data = wav_file.readframes(wav_file.getnframes())
        return base64.b64encode(pcm_data).decode("utf-8"), sample_rate


def build_two_speaker_training_request() -> TrainingRequest:
    """Build a training request from the example speaker WAV files."""
    speaker1_audio_data, speaker1_rate = read_audio_file_as_base64(
        EXAMPLE_DATA_DIR / "speaker1_1.wav"
    )
    speaker2_audio_data, speaker2_rate = read_audio_file_as_base64(
        EXAMPLE_DATA_DIR / "speaker2_1.wav"
    )

    return TrainingRequest(
        voice_samples=[
            VoiceSample(
                user="speaker1",
                audio=AudioInput(
                    audio_data=speaker1_audio_data,
                    sample_rate=speaker1_rate,
                ),
            ),
            VoiceSample(
                user="speaker2",
                audio=AudioInput(
                    audio_data=speaker2_audio_data,
                    sample_rate=speaker2_rate,
                ),
            ),
        ]
    )


def build_recognition_request(file_name: str) -> RecognitionRequest:
    """Build a recognition request from an example WAV file."""
    audio_data, sample_rate = read_audio_file_as_base64(EXAMPLE_DATA_DIR / file_name)
    return RecognitionRequest(
        audio=AudioInput(audio_data=audio_data, sample_rate=sample_rate)
    )


class FakeVoiceEncoder:
    """Fast deterministic encoder for API/client contract tests."""

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        """Return stable fake speaker classes for the bundled fixtures."""
        if np.std(wav) > 0.18:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


@pytest.fixture()
def fake_encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Patch the global recognizer to avoid the real Resemblyzer/Torch stack."""
    recognizer._reference_embeddings = {}
    recognizer._is_trained = False
    recognizer.embeddings_directory = str(tmp_path / "embeddings")
    monkeypatch.setattr(recognizer, "_encoder", FakeVoiceEncoder())

    yield

    recognizer._reference_embeddings = {}
    recognizer._is_trained = False


@pytest.fixture(scope="module")
def api_server():
    """Start API server for testing."""
    import time

    import httpx

    server_process = Process(target=start_api_server)
    server_process.start()

    # Wait for server to be ready with health checks
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = httpx.get(f"{API_BASE_URL}/health", timeout=1.0)
            if response.status_code == 200:
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt == max_attempts - 1:
                server_process.terminate()
                server_process.join(timeout=5)
                raise RuntimeError("Server failed to start within timeout")
            time.sleep(0.5)

    yield

    # Cleanup
    server_process.terminate()
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.kill()


@pytest.mark.asyncio
async def test_api_client_train_and_recognize_speakers(fake_encoder: None):
    """Test API/client contracts without the slow real ML encoder."""
    # Read training audio files
    speaker1_training_file = EXAMPLE_DATA_DIR / "speaker1_1.wav"
    speaker2_training_file = EXAMPLE_DATA_DIR / "speaker2_1.wav"

    # Read recognition audio files
    speaker1_recognition_file = EXAMPLE_DATA_DIR / "speaker1_2.wav"
    speaker2_recognition_file = EXAMPLE_DATA_DIR / "speaker2_2.wav"

    # Verify all files exist
    assert speaker1_training_file.exists(), f"Missing {speaker1_training_file}"
    assert speaker2_training_file.exists(), f"Missing {speaker2_training_file}"
    assert speaker1_recognition_file.exists(), f"Missing {speaker1_recognition_file}"
    assert speaker2_recognition_file.exists(), f"Missing {speaker2_recognition_file}"

    transport = httpx.ASGITransport(app=app)
    async with SpeakerRecognitionClient(
        "http://testserver",
        timeout=5.0,
        transport=transport,
    ) as client:
        health = await client.health_check()
        assert health.status == "healthy"

        training_result = await client.train(build_two_speaker_training_request())
        assert training_result.status == "success"
        assert training_result.count == 2
        assert "speaker1" in training_result.trained_users
        assert "speaker2" in training_result.trained_users

        recognition_result_1 = await client.recognize(
            build_recognition_request("speaker1_2.wav")
        )
        assert recognition_result_1.user_id == "speaker1", (
            f"Expected speaker1, got {recognition_result_1.user_id} "
            f"with confidence {recognition_result_1.confidence}. "
            f"All scores: {recognition_result_1.all_scores}"
        )

        recognition_result_2 = await client.recognize(
            build_recognition_request("speaker2_2.wav")
        )
        assert recognition_result_2.user_id == "speaker2", (
            f"Expected speaker2, got {recognition_result_2.user_id} "
            f"with confidence {recognition_result_2.confidence}. "
            f"All scores: {recognition_result_2.all_scores}"
        )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_stack_train_and_recognize_speakers(api_server: None):
    """Test the real HTTP server and Resemblyzer/Torch encoder stack."""
    # Read training audio files
    speaker1_training_file = EXAMPLE_DATA_DIR / "speaker1_1.wav"
    speaker2_training_file = EXAMPLE_DATA_DIR / "speaker2_1.wav"

    # Read recognition audio files
    speaker1_recognition_file = EXAMPLE_DATA_DIR / "speaker1_2.wav"
    speaker2_recognition_file = EXAMPLE_DATA_DIR / "speaker2_2.wav"

    # Verify all files exist
    assert speaker1_training_file.exists(), f"Missing {speaker1_training_file}"
    assert speaker2_training_file.exists(), f"Missing {speaker2_training_file}"
    assert speaker1_recognition_file.exists(), f"Missing {speaker1_recognition_file}"
    assert speaker2_recognition_file.exists(), f"Missing {speaker2_recognition_file}"

    async with SpeakerRecognitionClient(API_BASE_URL, timeout=300.0) as client:
        health = await client.health_check()
        assert health.status == "healthy"

        # Train the model
        training_result = await client.train(build_two_speaker_training_request())
        assert training_result.status == "success"
        assert training_result.count == 2
        assert "speaker1" in training_result.trained_users
        assert "speaker2" in training_result.trained_users

        # Test recognition for speaker1
        recognition_result_1 = await client.recognize(
            build_recognition_request("speaker1_2.wav")
        )
        assert recognition_result_1.user_id == "speaker1", (
            f"Expected speaker1, got {recognition_result_1.user_id} "
            f"with confidence {recognition_result_1.confidence}. "
            f"All scores: {recognition_result_1.all_scores}"
        )

        # Test recognition for speaker2
        recognition_result_2 = await client.recognize(
            build_recognition_request("speaker2_2.wav")
        )
        assert recognition_result_2.user_id == "speaker2", (
            f"Expected speaker2, got {recognition_result_2.user_id} "
            f"with confidence {recognition_result_2.confidence}. "
            f"All scores: {recognition_result_2.all_scores}"
        )
