import os
import tqdm
from models import WhisperCPP, OpenAIAPI, apply_speedup_and_silence_removal


def main():
    # --- Configuration ---
    models_to_test = {
        "whisper_cpp_small": WhisperCPP(),
        "openai_whisper_1": OpenAIAPI(model="whisper-1"),
        "openai_gpt_4o": OpenAIAPI(model="gpt-4o-transcribe"),
        "openai_gpt_4o_mini": OpenAIAPI(model="gpt-4o-mini-transcribe"),
    }
    speedup_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    data_dir = "data"
    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)

    # --- Evaluation Loop ---
    for audio_file in os.listdir(data_dir):
        if not audio_file.endswith(".wav"):
            continue

        input_path = os.path.join(data_dir, audio_file)
        print(f"--- Processing: {input_path} ---")

        for speed in tqdm.tqdm(speedup_factors, desc="Speedup Factors"):
            processed_audio_path = os.path.join(
                results_dir, f"{os.path.splitext(audio_file)[0]}_speed{speed}.wav"
            )

            # Apply audio processing
            apply_speedup_and_silence_removal(input_path, processed_audio_path, speed)

            # Transcribe with each model
            for model_name, model in models_to_test.items():
                try:
                    transcript = model.transcribe(processed_audio_path)
                    result = f"Model: {model_name}, Speed: {speed}x, Transcript: {transcript}"
                    print(result)

                    # Save result to file
                    with open(os.path.join(results_dir, "results.txt"), "a") as f:
                        f.write(result + "\n")

                except Exception as e:
                    print(f"Error with model {model_name} at speed {speed}: {e}")


if __name__ == "__main__":
    main()
