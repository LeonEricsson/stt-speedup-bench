# Does audio speedup affect speech recognition?

This project contains code for running experiments on the robustness of (popular) speech-recognition models to sped-up audio. 

The core idea is to take a standard speech-to-text (STT) evaluation set, create copies of it at various speed factors, and then measure the degradation in transcription accuracy (WER/CER).

## Experiment


### Initial experiments
Initial experiments were conducted on the Whisper model family. The FLEURS dataset was chosen for evaluation, specifically the test sets in English, Spanish, and Swedish to assess robustness across a small variety of languages. The evaluation metrics used were Word Error Rate (WER) and Character Error Rate (CER), calculated using [jiwer](https://github.com/jitsi/jiwer). The tested speedup factors were: 1.0, 1.5, 2.0, 2.5, and 3.0.

An overview of the results is presented in the figure below.

<p align="center">
  <img src="/results/assets/error_rate_speedup-1.png" alt="Error Rate vs Speedup" width="65%">
</p>

<p align="center">
  <em>The error rate across three languages in the FLEURS test set, at increasing speedup factors. Error rates are averaged across language.</em>
</p>

---

The detailed results are given in tables below.
<p align="center">
  <img src="/results/assets/whisper-large-v3-turbo.png" width="23%">
  <img src="/results/assets/whisper-medium.png" width="23%">
  <img src="/results/assets/whisper-small.png" width="23%">
  <img src="/results/assets/gpt-4o-transcribe.png" width="23%">
</p>

### Finer resolution
Given the exponential nature of the error rate I conduct more fine grained experiments around speedup factors 1.0 - 1.6. Otherwise the experiment is held the same (FLEURS test set, evaluation metrics, languages, ...).

<p align="center">
  <img src="/results/assets/error_rate_speedup-fine.png" alt="Error Rate vs Speedup" width="65%">
</p>

---

The detailed results are given in tables below.
<p align="center">
  <img src="/results/assets/whisper-large-v3-turbo-fine.png" width="30%">
  <img src="/results/assets/whisper-medium-fine.png" width="30%">
  <img src="/results/assets/whisper-small-fine.png" width="30%">
</p>


## Recreating the results

### 1. Set up the environment

This project uses `uv` for package management.

```bash
# install python dependencies
uv sync
```

### 2. Download and preprocess the data

We use the [FLEURS dataset](https://huggingface.co/datasets/google/fleurs) for evaluation. The `download_dataset.py` script will download the necessary splits.

```bash
uv run scripts/download_dataset.py
```

This will download the data to `data/fleurs`.

Next, we need to preprocess the data to create the sped-up versions.

```bash
uv run scripts/preprocess_fleurs.py
```

This script takes the downloaded FLEURS splits from `data/fleurs` and creates a new version at `data/fleurs_preprocessed`. This new version contains multiple copies of each utterance, sped up by factors of 1.0, 1.5, 2.0, 2.5, and 3.0.

### 3. Run the evaluation

The `src/evaluate_fleurs.py` script runs the evaluation. It takes two main arguments: `--interface` and `--model-id`.

The interface determines how to run the model:

*   `whisper_transformers`: Uses the Hugging Face `transformers` library. This is the easiest way to run open-weight Whisper models.
*   `whisper_cpp`: Uses a local `whisper.cpp` server. This is faster for CPU inference.
*   `openai`: Uses the OpenAI API for closed-source models like `whisper-large-v3`.

The `model-id` specifies the model to use, e.g., `openai/whisper-large-v3`.

Here are some example commands:

```bash
# Evaluate Whisper large-v3 using the transformers library
uv run src/evaluate_fleurs.py --interface whisper_transformers --model-id openai/whisper-large-v3

# Evaluate Whisper medium using whisper.cpp
# (make sure the whisper.cpp server is running first)
uv run src/evaluate_fleurs.py --interface whisper_cpp --model-id large-v3

# Evaluate OpenAI's API-based Whisper
uv run src/evaluate_fleurs.py --interface openai --model-id whisper-1
```

The results will be saved as CSV files in the `results/` directory.

## Caveats and Limitations

*   **Speed-up method**: The audio is sped up using `torchaudio.sox_effects` with the `tempo` effect. This method preserves pitch, but may introduce artifacts that are not representative of natural fast speech.
*   **Dataset**: I have only tested on the test set of FLEURS.
*   **Languages**: The evaluation is currently run on English, Spanish, and Swedish.
*   **Normalization**: The reference and hypothesis texts are normalized before calculating WER/CER. The normalization includes lowercasing, removing punctuation, and removing Kaldi non-words. This is a standard practice, but it can hide certain types of errors.