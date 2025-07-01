# evaluate_fleurs.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk, Audio
from tqdm import tqdm
from evaluate import load
import jiwer, unicodedata

from models import TransformersWhisper, WhisperCPP, OpenAIAPI

from transformers.pipelines.pt_utils import KeyDataset

MODEL_REGISTRY = {
    "whisper_transformers": TransformersWhisper,
    "whisper_cpp": WhisperCPP,
    "openai": OpenAIAPI,
}

DATA_DIR = Path("data/fleurs_preprocessed_fine")
RESULTS_DIR = Path("results")
LANGUAGES = ["en_us", "es_419", "sv_se"]
BATCH_SIZE = 64
LANG_MAP = {
    "en_us": "english",
    "es_419": "spanish",
    "sv_se": "swedish",
}

METRIC_WER = load("wer")
METRIC_CER = load("cer")

NORMALISE = jiwer.Compose(
    [
        jiwer.RemoveKaldiNonWords(),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToSingleSentence(),
    ]
)


def norm(txt: str) -> str:
    return NORMALISE(unicodedata.normalize("NFC", txt))


def corpus_and_macro(preds, refs):
    wer_micro = METRIC_WER.compute(predictions=preds, references=refs)
    cer_micro = METRIC_CER.compute(predictions=preds, references=refs)

    wer_per = [
        METRIC_WER.compute(predictions=[p], references=[r]) for p, r in zip(preds, refs)
    ]
    cer_per = [
        METRIC_CER.compute(predictions=[p], references=[r]) for p, r in zip(preds, refs)
    ]

    return (
        wer_micro,
        cer_micro,
        sum(wer_per) / len(wer_per),
        sum(cer_per) / len(cer_per),
    )


def load_dataset(lang: str):
    ds = load_from_disk(str(DATA_DIR / lang))
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def evaluate_lang(lang: str, model, writer: csv.DictWriter, limit: int | None) -> None:
    ds = load_dataset(lang)
    if limit:
        ds = ds.select(range(limit))

    # wrap the HF dataset so pipeline can consume its "audio" column directly
    kd = KeyDataset(ds, "audio")

    # run pipeline once, internally batched
    out = list(
        tqdm(
            model.pipe(
                kd,
                generate_kwargs={**model.generate_kwargs, "language": LANG_MAP[lang]},
            ),
            total=len(ds),
            desc=f"{lang} eval",
        )
    )

    # collect references, predictions, and speedups
    refs = [norm(r) for r in ds["transcription"]]
    preds = [norm(o["text"]) for o in out]
    speedups = ds["speedup"]

    # slice by speedup factor
    for f in sorted(set(speedups)):
        idxs = [i for i, s in enumerate(speedups) if s == f]
        p_sl = [preds[i] for i in idxs]
        r_sl = [refs[i] for i in idxs]
        werμ, cerμ, werM, cerM = corpus_and_macro(p_sl, r_sl)
        writer.writerow(
            {
                "language": lang,
                "speedup": f,
                "n_utts": len(idxs),
                "wer_micro": werμ,
                "cer_micro": cerμ,
                "wer_macro": werM,
                "cer_macro": cerM,
            }
        )


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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples per language to this amount.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    model_cls = MODEL_REGISTRY[args.interface]
    model = model_cls(model_id=args.model_id)
    out_file = RESULTS_DIR / f"fleurs_{args.interface}.csv"

    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "speedup",
                "n_utts",
                "wer_micro",
                "cer_micro",
                "wer_macro",
                "cer_macro",
            ],
        )
        writer.writeheader()

        for lang in LANGUAGES:
            if not (DATA_DIR / lang).exists():
                print(f"[skip] {lang}: {DATA_DIR / lang} not found")
                continue
            evaluate_lang(lang, model, writer, args.limit)

    print(f"✅ Results written to {out_file}")


if __name__ == "__main__":
    main()
