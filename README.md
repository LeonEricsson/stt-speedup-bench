# Cheaper Transcriptions, Pricer Errors

Last weekend I stumbled across a post on X that pointed to George Mandis’ write-up, [**“OpenAI Charges by the Minute, So Make the Minutes Shorter.”**](https://george.mand.is/2025/06/openai-charges-by-the-minute-so-make-the-minutes-shorter/) In it, George doubles the playback speed of a YouTube clip, feeds the audio to a speech-to-text model, and still gets a passable transcript—good enough for an LLM to crank out a coherent summary. He openly admits the test isn’t rigorous and that he cares more about summary fidelity than word-for-word accuracy. But it was enough to peak my interest; naturally, it sounded too good to be true. I figured that in George's case, the language model was able to conjure up a decent-looking summary even if the transcription was poor. Still, even a minor speedup could lead to significant savings in inference costs, so I decided to run a more disciplined experiment. Also, it gave me an opportunity to try out gemini-cli. 

This repo benchmarks the Whisper model family and GPT 4o on tempo-scaled inputs drawn from the multilingual FLEURS corpus. Playback rates step from 1.0× (baseline) all the way up to 3.0×. For each setting I compute macro+micro, Word Error Rate (WER) + Character Error Rate (CER) and watch how fast they drift north. We know performance will deteriorate, what's interesting here is mapping that degradation curve, is there a worthwhile trade-off somewhere along said curve?

### TL;DR

Results compiled into a single figure. Performance degradation is exponential, at 2× playback most models are already 3–5× worse; push to 2.5× and accuracy falls off a cliff, with 20× degradation not uncommon. There are still sweet spots, though: Whisper-large-turbo only drifts from 5.39 % to 6.92 % WER (≈ 28 % relative hit) at 1.5×, and GPT-4o tolerates 1.2 × with a trivial ~3 % penalty.

<p align="center">
  <img src="results/assets/tldr.png" alt="Error Rate vs Speedup" width="70%">
</p>

<p align="center">
  <em>The word error rate across three languages in the FLEURS test set, at increasing speedup factors. Error rates are averaged across language.</em>
</p>

tip: remove silences. that's the real 0 performance loss trick.

## Experiment

### Coarse
Experiments targeted the Whisper-model family (small, medium, and large-v3-turbo) and GPT-4o Transcribe, using the multilingual FLEURS test sets (English, Spanish, and Swedish). Transcription accuracy was assessed using Word Error Rate (WER) and Character Error Rate (CER), computed via [jiwer](https://github.com/jitsi/jiwer). Tested speedup factors included: 1.0, 1.5, 2.0, 2.5, and 3.0.

Results summary:

<p align="center">
  <img src="results/assets/error_rate_speedup-1.png" alt="Error Rate vs Speedup" width="71%">
</p>

Detailed per-model results:
<p align="center">
  <img src="results/assets/whisper-large-v3-turbo.png" width="23%">
  <img src="results/assets/whisper-medium.png" width="23%">
  <img src="results/assets/whisper-small.png" width="23%">
  <img src="results/assets/gpt-4o-transcribe.png" width="23%">
</p>

### Finer resolution
It's glaringly evident that larger speedup factors are a no-go when measuring word for word transcription accuracy, however there may be a trade-off worth considering at the 1.0 - 1.5 range, so let's take a closer look at that. Conditions and evaluation protocols remained consistent with coarse tests.


<p align="center">
  <img src="results/assets/error_rate_speedup-fine.png" alt="Error Rate vs Speedup" width="65%">
</p>

Detailed per-model results:
<p align="center">
  <img src="results/assets/whisper-large-v3-turbo-fine.png" width="23%">
  <img src="results/assets/whisper-medium-fine.png" width="23%">
  <img src="results/assets/whisper-small-fine.png" width="23%">
  <img src="results/assets/gpt-4o-transcribe-fine.png" width="23%">
</p>

### words per minute 

Currently, the study uniformly applies speedup factors to audio samples, without explicitly controlling or measuring Words per Minute (WPM). However, speaker-dependent variability in baseline WPM could significantly influence model robustness to speed-induced artifacts. I'm considering
decoupling speed-up factors from samples and instead plotting performance against raw WPM. 

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

## Caveats

*   **Speed-up method**: The audio is sped up using `torchaudio.sox_effects` with the `tempo` effect. This method preserves pitch, but may introduce artifacts that are not representative of natural fast speech.
*   **Dataset**: I have only tested on the test set of FLEURS.
*   **Languages**: The evaluation is currently run on English, Spanish, and Swedish.
*   **Normalization**: The reference and hypothesis texts are normalized before calculating WER/CER. The normalization includes lowercasing, removing punctuation, and removing Kaldi non-words. This is a standard practice, but it can hide certain types of errors.