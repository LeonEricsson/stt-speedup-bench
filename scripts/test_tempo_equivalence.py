import torch
import torchaudio
from datasets import load_from_disk, Audio
import numpy as np
import subprocess
import os

# Configuration
ARROW_FILE = "/home/leon/git/leon/stt-speedup/data/fleurs/en_us"
OUTPUT_DIR = "/home/leon/git/leon/stt-speedup/temp_audio_test"
TEMP_ORIGINAL_WAV = os.path.join(OUTPUT_DIR, "original.wav")
TEMP_SOX_WAV = os.path.join(OUTPUT_DIR, "sox_tempo.wav")
TEMP_FFMPEG_WAV = os.path.join(OUTPUT_DIR, "ffmpeg_atempo.wav")
SPEEDUP_FACTOR = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_test():
    # 1. Load dataset and extract an audio sample
    print("Loading dataset...")
    ds = load_from_disk(ARROW_FILE)
    # Ensure audio is loaded with a specific sampling rate if not already
    # ds = ds.cast_column(
    #     "audio", Audio(sampling_rate=16000)
    # )  # Assuming 16kHz, adjust if needed

    sample = ds[36]  # Get the first sample
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    print(
        f"Original audio sample loaded. Shape: {audio_array.shape}, SR: {sampling_rate}"
    )

    # Save original audio to a temporary WAV file for FFmpeg
    torchaudio.save(
        TEMP_ORIGINAL_WAV, torch.from_numpy(audio_array).unsqueeze(0), sampling_rate
    )
    print(f"Original audio saved to {TEMP_ORIGINAL_WAV}")

    # 2. Apply tempo change using torchaudio.sox_effects
    print("Applying tempo with torchaudio.sox_effects...")
    wav_tensor = torch.as_tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    sox_wav_tensor, _ = torchaudio.sox_effects.apply_effects_tensor(
        wav_tensor, sampling_rate, effects=[["tempo", f"{SPEEDUP_FACTOR}"]]
    )
    torchaudio.save(TEMP_SOX_WAV, sox_wav_tensor, sampling_rate)
    print(f"SoX tempo audio saved to {TEMP_SOX_WAV}")

    # 3. Apply tempo change using ffmpeg atempo
    print("Applying tempo with ffmpeg atempo...")
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-i",
        TEMP_ORIGINAL_WAV,
        "-filter:a",
        f"atempo={SPEEDUP_FACTOR}",
        TEMP_FFMPEG_WAV,
    ]
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True)
        print(f"FFmpeg atempo audio saved to {TEMP_FFMPEG_WAV}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        return

    # 4. Compare the resulting audio data
    print("Comparing audio files...")
    sox_loaded, sox_sr = torchaudio.load(TEMP_SOX_WAV)
    ffmpeg_loaded, ffmpeg_sr = torchaudio.load(TEMP_FFMPEG_WAV)

    print(f"SoX output shape: {sox_loaded.shape}, SR: {sox_sr}")
    print(f"FFmpeg output shape: {ffmpeg_loaded.shape}, SR: {ffmpeg_sr}")

    # Check sampling rates
    if sox_sr != ffmpeg_sr:
        print("Sampling rates differ!")
        return

    # Pad the shorter audio to match the longer one for comparison
    max_len = max(sox_loaded.shape[1], ffmpeg_loaded.shape[1])
    sox_padded = torch.nn.functional.pad(sox_loaded, (0, max_len - sox_loaded.shape[1]))
    ffmpeg_padded = torch.nn.functional.pad(
        ffmpeg_loaded, (0, max_len - ffmpeg_loaded.shape[1])
    )

    # Compare using a numerical tolerance due to potential floating point differences
    # and slight algorithmic variations
    tolerance = 1e-6  # Adjust tolerance as needed
    are_close = torch.allclose(sox_padded, ffmpeg_padded, atol=tolerance)

    if are_close:
        print(f"The audio outputs are numerically close (tolerance={tolerance}).")
        print(
            "This suggests torchaudio.sox_effects tempo and ffmpeg atempo are functionally equivalent."
        )
    else:
        print(f"The audio outputs are NOT numerically close (tolerance={tolerance}).")
        print("This might indicate differences in implementation or precision.")
        # You might want to inspect differences further, e.g., by plotting or listening
        diff = torch.abs(sox_padded - ffmpeg_padded).max()
        print(f"Maximum absolute difference: {diff}")

    # Clean up temporary files
    # os.remove(TEMP_ORIGINAL_WAV)
    # os.remove(TEMP_SOX_WAV)
    # os.remove(TEMP_FFMPEG_WAV)
    # os.rmdir(OUTPUT_DIR) # Only if directory is empty


if __name__ == "__main__":
    run_test()
