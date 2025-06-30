"""Evaluate STT models on the preprocessed FLEURS dataset.

This script expects the dataset to be downloaded with ``download_dataset.py`` and
preprocessed with ``preprocess_fleurs.py``. The preprocessing step creates
speed‑up variants of each clip and stores them in ``data/fleurs_preprocessed``.

Usage:
    uv run scripts/evaluate_fleurs.py [--model MODEL]
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Dict, Any, List
import tempfile

from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk, Audio
import soundfile as sf
from evaluate import load
import jiwer, unicodedata

from models import TransformersWhisper, WhisperCPP, OpenAIAPI

MODEL_REGISTRY = {
    "whisper_transformers": TransformersWhisper,
    "whisper_cpp": WhisperCPP,
    "openai": OpenAIAPI,
}

DEFAULT_MODEL_ID = {
    "whisper_transformers": "openai/whisper-large-v3-turbo",
    "openai": "gpt-4o-transcribe",
}

# ---------------------------- configuration ---------------------------------
DATA_DIR = Path("data/fleurs_preprocessed")
RESULTS_DIR = Path("results")
LANGUAGES = ["en_us", "es_419", "sv_se"]
BATCH_SIZE = 64
LANG_MAP = {
    "en_us": "english",
    "es_419": "spanish",
    "sv_se": "swedish",
}

# ---- metric objs once ----
METRIC_WER = load("wer")
METRIC_CER = load("cer")

# ---- jiwer normalisation pipeline ----
NORMALISE = jiwer.Compose([
    jiwer.RemoveKaldiNonWords(),       # strips [noise] <sil> …
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),         # keep apostrophes? drop? adjust if needed
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToSingleSentence(),    # optional – kills trailing space before OOV
])

def norm(txt: str) -> str:
    """Unicode NFC + jiwer pipeline."""
    return NORMALISE(unicodedata.normalize("NFC", txt))

def corpus_and_macro(preds, refs):
    """Return (wer_micro, cer_micro, wer_macro, cer_macro)."""
    wer_micro = METRIC_WER.compute(predictions=preds, references=refs)
    cer_micro = METRIC_CER.compute(predictions=preds, references=refs)

    wer_per = [METRIC_WER.compute(predictions=[p], references=[r])
               for p, r in zip(preds, refs)]
    cer_per = [METRIC_CER.compute(predictions=[p], references=[r])
               for p, r in zip(preds, refs)]

    return wer_micro, cer_micro, sum(wer_per)/len(wer_per), sum(cer_per)/len(cer_per)

def collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare a batch for inference."""
    audio_arrays = [sample["audio"]["array"] for sample in batch]
    transcripts = [sample.get("transcription") or sample.get("raw_transcription") for sample in batch]
    speedups = [sample["speedup"] for sample in batch]
    languages = [sample.get("language") for sample in batch]
    return {
        "audio": audio_arrays,
        "text": transcripts,
        "speedup": speedups,
        "language": languages,
    }


def load_dataset(lang: str):
    """Return a Dataset object cast to Audio with 16 kHz sampling rate."""
    ds = load_from_disk(str(DATA_DIR / lang))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    return ds


def transcribe_batch(model, audios: List, lang_code: str) -> List[str]:
    """Transcribe a list of audio arrays using the provided model."""
    if isinstance(model, TransformersWhisper):
        return model.transcribe(audios, language=LANG_MAP.get(lang_code))
    else:
        texts = []
        for wav in audios:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, wav, 16000)
                texts.append(model.transcribe(tmp.name))
        return texts


def evaluate_lang(lang: str, model, writer: csv.DictWriter) -> None:
    """Infer once, then metric per-speed slice."""
    ds = load_dataset(lang)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # --- gather everything first ---
    refs, preds, speedups = [], [], []

    for batch in tqdm(loader, desc=f"{lang} eval"):
        batch_preds = transcribe_batch(model, batch["audio"], lang)
        refs     += [norm(t) for t in batch["text"]]
        preds    += [norm(t) for t in batch_preds]
        speedups += batch["speedup"]

    # --- slice by speedup factor ---
    factors = sorted(set(speedups))      
    for f in factors:
        idx  = [i for i, s in enumerate(speedups) if s == f]
        p_sl = [preds[i] for i in idx]
        r_sl = [refs[i]  for i in idx]

        werμ, cerμ, werM, cerM = corpus_and_macro(p_sl, r_sl)
        writer.writerow({
            "language":   lang,
            "speedup":    f,
            "n_utts":     len(idx),
            "wer_micro":  werμ,
            "cer_micro":  cerμ,
            "wer_macro":  werM,
            "cer_macro":  cerM,
        })


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FLEURS")
    parser.add_argument(
        "--interface",
        choices=MODEL_REGISTRY.keys(),
        default="whisper_transformers",
        help="STT interface implementation to use",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Underlying model identifier (e.g. openai/whisper-large-v3-turbo)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    model_class = MODEL_REGISTRY[args.interface]
    model_id = args.model_id or DEFAULT_MODEL_ID.get(args.interface)
    model = model_class(model_id=model_id)
    out_file = RESULTS_DIR / f"fleurs_{args.interface}.csv"

    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "speedup", "n_utts", "wer_micro", "cer_micro", "wer_macro", "cer_macro"],   
        )
        writer.writeheader()
        for lang in LANGUAGES:
            path = DATA_DIR / lang
            if not path.exists():
                print(f"[skip] {lang}: {path} not found")
                continue
            evaluate_lang(lang, model, writer)

    print(f"Results written to {out_file}")


if __name__ == "__main__":
    main()