#!/usr/bin/env python
"""
filter_fleurs.py – Filter utterances longer than MAX_SECONDS.
"""

from pathlib import Path
from datasets import load_from_disk, Audio

DATA_DIR = Path("data/fleurs")  # original
OUT_DIR = Path("data/fleurs_filtered")  # destination
LANGUAGES = ["en_us", "es_419", "sv_se"]
SPLIT = "test"
MAX_SECONDS = 25.0


def shorter_than_max(ex):
    dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
    return dur < MAX_SECONDS


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang in LANGUAGES:
        src = DATA_DIR / lang
        if not src.exists():
            print(f"[skip] {lang}: {src} missing")
            continue

        print(f"→ {lang}: loading")
        ds = load_from_disk(str(src)).cast_column("audio", Audio(sampling_rate=None))

        pruned = ds.filter(
            shorter_than_max,  # <— predicate only
            batched=False,
            load_from_cache_file=False,  # avoid stale caches
            keep_in_memory=False,
        )

        dst = OUT_DIR / lang
        pruned.save_to_disk(str(dst))
        print(f"   kept {len(pruned)} / {len(ds)} samples  ➜  {dst}")


if __name__ == "__main__":
    main()
