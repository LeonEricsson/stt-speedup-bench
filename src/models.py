import os
import requests
import openai
import subprocess
from dotenv import load_dotenv
from typing import List, Iterator, Dict
import soundfile as sf
import tempfile

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

load_dotenv()


class WhisperCPP:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.url = f"http://{host}:{port}/inference"

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            response = requests.post(self.url, files=files)
            response.raise_for_status()
            return response.json()["text"]


class OpenAIAPI:
    def __init__(
        self, model_id: str = "whisper-1", api_key: str | None = None, **kwargs
    ):
        self.model = model_id
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("LOCAL_OPENAI_API_KEY"),
        )
        self.generate_kwargs = {}

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
            )
        return transcript.text

    def pipe(self, dataset_iterator, **kwargs) -> Iterator[Dict[str, str]]:
        for item in dataset_iterator:
            # item is a dict like {'array': np.array(...), 'sampling_rate': 16000}
            audio_array = item["array"]
            sampling_rate = item["sampling_rate"]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                sf.write(tmpfile.name, audio_array, sampling_rate)
                try:
                    text = self.transcribe(tmpfile.name)
                    yield {"text": text}
                except Exception as e:
                    print(f"Error transcribing file: {e}")
                    raise e


class TransformersWhisper:
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        batch_size: int = 16,
        generate_kwargs: dict = None,
    ):
        # device & dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        # load model & processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # build pipeline
        pipe_kwargs = {
            "model": self.model,
            "tokenizer": self.processor.tokenizer,
            "feature_extractor": self.processor.feature_extractor,
            "torch_dtype": self.torch_dtype,
            "device": 0 if "cuda" in self.device else -1,
        }

        self.pipe = pipeline("automatic-speech-recognition", **pipe_kwargs)

        # optional decoding/generation tweaks
        self.generate_kwargs = generate_kwargs or {
            # you can override these defaults when calling transcribe()
            "max_new_tokens": 440,
        }

    def transcribe(self, audios: List, **override_generate_kwargs) -> str:
        """
        Transcribe audio batch
        :param audio_path: batch of audio arrays
        :param override_generate_kwargs: any generate_kwargs to override defaults
        :returns: transcript text
        """
        # merge defaults with overrides
        gen_kwargs = {**self.generate_kwargs, **override_generate_kwargs}
        preds = self.pipe(audios, generate_kwargs=gen_kwargs)
        return [p["text"] for p in preds]


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
