import os
import requests
import openai
import subprocess
from abc import ABC, abstractmethod


class STTModel(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass


class WhisperCPP(STTModel):
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.url = f"http://{host}:{port}/inference"

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            response = requests.post(self.url, files=files)
            response.raise_for_status()
            return response.json()["text"]


class OpenAIAPI(STTModel):
    def __init__(self, model: str = "whisper-1", api_key: str | None = None):
        self.model = model
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            transcript = openai.Audio.transcribe(self.model, f)
        return transcript["text"]


def apply_speedup_and_silence_removal(input_path: str, output_path: str, speed: float):
    """
    Apply speedup and silence removal to an audio file using ffmpeg.
    """
    atempo_filter = f"atempo={speed}"
    silenceremove_filter = "silenceremove=start_periods=1:start_duration=0.2:start_threshold=-50dB:stop_periods=-1:stop_duration=1:stop_threshold=-50dB"

    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-af",
        f"{silenceremove_filter},{atempo_filter}",
        "-ar",
        "16000",
        output_path,
        "-y",
    ]
    subprocess.run(command, check=True, capture_output=True)
