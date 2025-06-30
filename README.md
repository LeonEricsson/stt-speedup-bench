# STT Speedup Evaluation

This project evaluates the performance of various speech-to-text (STT) models when the input audio is sped up by different factors.

## Project Goal

The primary goal is to measure how the accuracy of STT models, such as Whisper and OpenAI's models, is affected by increasing the speed of the input audio. This is achieved by programmatically speeding up audio files from an evaluation dataset and comparing the transcription results against the ground truth.

## Features

*   **Audio Speed-up:** Utilizes `ffmpeg` to apply the `atempo` filter for speeding up audio files.
*   **Silence Removal:** Uses `ffmpeg`'s `silenceremove` filter to remove silences from the audio, which can also affect STT performance.
*   **Multi-model Evaluation:** Supports evaluating different STT models:
    *   **Whisper:** Runs locally using `whisper.cpp` as a server.
    *   **OpenAI API:** Interacts with OpenAI's transcription API (e.g., for `whisper-4o`).
*   **Extensible:** The project is designed to be easily extendable to include other STT models and evaluation datasets.

## Getting Started

### Prerequisites

*   Python 3.8+
*   `ffmpeg`
*   An OpenAI API key (if using OpenAI models)
*   `whisper.cpp` (for running Whisper models locally)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd stt-speedup
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up `whisper.cpp`:**
    Follow the instructions in the [whisper.cpp repository](https://github.com/ggerganov/whisper.cpp) to build and run the server.

### Usage

1.  **Prepare your evaluation dataset:**
    Place your audio files in the `data/` directory.

2.  **Run the evaluation:**
    ```bash
    python main.py
    ```

    The script will process the audio files, send them to the configured STT models, and save the results in the `results/` directory.

## Project Structure

```
.
├── data/
│   └── dummy.wav
├── results/
├── scripts/
│   └── run_whisper_server.sh
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── models.py
├── .gitignore
├── pyproject.toml
└── README.md
```

*   `data/`: Contains the input audio files for evaluation.
*   `results/`: Stores the transcription results from the STT models.
*   `scripts/`: Helper scripts, such as for running the `whisper.cpp` server.
*   `src/`: The main source code for the project.
    *   `main.py`: The main script that orchestrates the evaluation process.
    *   `models.py`: Contains the logic for interacting with the different STT models.
*   `pyproject.toml`: Python project configuration.
*   `README.md`: This file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
