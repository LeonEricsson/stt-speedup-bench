import os
from datasets import load_dataset


def main():
    # --- Configuration ---
    languages = ["en_us", "es_419", "sv_se"]
    data_dir = "data/fleurs"

    os.makedirs(data_dir, exist_ok=True)

    # --- Download Loop ---
    for lang in languages:
        print(f"--- Downloading: {lang} ---")
        dataset = load_dataset("google/fleurs", lang, split="test")
        dataset.save_to_disk(os.path.join(data_dir, lang))


if __name__ == "__main__":
    main()
