"""
duration_stats.py – reveal basic duration statistics for local FLEURS splits
"""

from pathlib import Path
from typing import Dict

import numpy as np
from datasets import load_from_disk, Audio


# ---- Configuration ---------------------------------------------------------
DATA_DIR = Path("data/fleurs")  # where you saved the datasets
LANGUAGES = ["en_us", "es_419", "sv_se"]
SPLIT = "test"
# ----------------------------------------------------------------------------


def summarise(durations: np.ndarray) -> Dict[str, float]:
    """Return a small dictionary of descriptive stats."""
    return {
        "count": int(durations.size),
        "min": float(durations.min()),
        "max": float(durations.max()),
        "mean": float(durations.mean()),
        "median": float(np.median(durations)),
        "p90": float(np.percentile(durations, 90)),
        "p95": float(np.percentile(durations, 95)),
        "p99": float(np.percentile(durations, 99)),
        "over_30s": int((durations > 30).sum()),
    }


def get_durations(ds) -> np.ndarray:
    """
    Cast the `audio` column to actual audio and compute each clip’s
    length in seconds = len(waveform) / sampling_rate.
    """
    ds = ds.cast_column("audio", Audio(sampling_rate=None))
    return np.array(
        [len(a["array"]) / a["sampling_rate"] for a in ds["audio"]],
        dtype=np.float32,
    )


def main() -> None:
    for lang in LANGUAGES:
        dset_path = DATA_DIR / lang
        if not dset_path.exists():
            print(f"[skip] {lang}: directory {dset_path} not found")
            continue

        print(f"\n=== {lang} · {SPLIT} split ===")
        ds = load_from_disk(str(dset_path))

        durations = get_durations(ds)
        stats = summarise(durations)

        for k, v in stats.items():
            # nice formatting: seconds to 3 decimals, ints unaltered
            print(f"{k:>9}: {v:.3f}" if isinstance(v, float) else f"{k:>9}: {v}")


if __name__ == "__main__":
    main()
