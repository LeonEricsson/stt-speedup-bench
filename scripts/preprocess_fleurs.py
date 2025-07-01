#!/usr/bin/env python
"""
preprocess_fleurs.py
────────────────────
• Input directory : data/fleurs             (output of your original download)
• Output directory: data/fleurs_preprocessed
  └─ LANG/test.arrow  (1×-, 1.5×-, 2×- speed versions, ≤100 utterances)

Run with:   uv run preprocess_fleurs.py
"""

from pathlib import Path
import random
from typing import Dict, Any

import torch
import torchaudio
import numpy as np
from datasets import load_from_disk, Audio, Dataset

# --------------------------- configuration ----------------------------------
IN_DIR = Path("data/fleurs")  # original test splits
OUT_DIR = Path("data/fleurs_preprocessed")  # final destination
LANGS = ["en_us", "es_419", "sv_se"]  # language IDs
MAX_SECONDS = 25.0  # duration ceiling
SPEEDUPS = [1.0, 1.5, 2.0, 2.5, 3.0]  # tempo factors
SEED = 42  # reproducibility
NUM_PROC = 4  # parallel workers
# ----------------------------------------------------------------------------


def shorter_than_max(example: Dict[str, Any]) -> bool:
    """True ↦ keep row (duration < MAX_SECONDS)."""
    dur = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return dur < MAX_SECONDS


def tempo_batch(batch):
    # Prepare output dict with the same keys the batch has
    out = {k: [] for k in batch}  # e.g. "text", "path", "id", …
    out["speedup"] = []  # new column

    for wav_dict, *others in zip(
        batch["audio"], *[batch[k] for k in batch if k != "audio"]
    ):
        wav0 = torch.as_tensor(wav_dict["array"], dtype=torch.float32).unsqueeze(
            0
        )  # [1, T]

        sr = wav_dict["sampling_rate"]

        for f in SPEEDUPS:
            if f == 1.0:
                wav = wav0
            else:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav0, sr, effects=[["tempo", f"{f}"]]
                )

            # audio column
            out["audio"].append({"array": wav.squeeze(0).numpy(), "sampling_rate": sr})

            # copy the rest straight through
            for name, value in zip([k for k in batch if k != "audio"], others):
                out[name].append(value)

            out["speedup"].append(f)

    return out


def preprocess_lang(lang: str) -> None:
    src = IN_DIR / lang
    if not src.exists():
        print(f"[skip] {lang}: {src} missing")
        return

    print(f"\n=== {lang} ===")

    # 1. load + materialise audio
    ds: Dataset = load_from_disk(str(src)).cast_column(
        "audio", Audio(sampling_rate=None)
    )

    # 2. drop long clips
    ds = ds.filter(
        shorter_than_max,
        num_proc=NUM_PROC,
        load_from_cache_file=False,
        desc="filter < MAX_SECONDS",
    )
    print(f"  kept {len(ds)} after duration filter")

    # 3. duplicate with speedups
    ds = ds.map(
        tempo_batch,  # returns *3×* rows per input
        batched=True,  # allows list outputs per field
        batch_size=32,
        num_proc=4,
        load_from_cache_file=False,
        remove_columns=ds.column_names,  # drop originals, keep augmented cols
        desc=f"{lang} tempo duplication",
    )
    print(f"  after duplication: {len(ds)} rows")

    # 4. save
    out_path = OUT_DIR / lang
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    print(f"  → wrote {out_path}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        preprocess_lang(lang)


if __name__ == "__main__":
    main()
