# Music maker

A small local web app for generating short tracks with `facebook/musicgen-small` and a PEFT LoRA adapter. The interface is plain HTML and CSS, and the app uses Python's built-in web server; it does not use Streamlit.

## Requirements

- Python 3.10 or 3.11
- The adapter folder from your training run, containing `adapter_config.json` and `adapter_model.safetensors` (or `.bin`)
- A compatible PyTorch installation; an NVIDIA GPU is recommended

## Setup

Open the `MusicGen-AI` folder in VS Code. In Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the command recommended by the official PyTorch installation selector for your OS and CUDA version, then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

## Configure the adapter

Copy `.env.example` to `.env`. Set `LORA_ADAPTER_PATH` to the adapter directory:

```env
LORA_ADAPTER_PATH=.
MUSICGEN_MODEL_ID=facebook/musicgen-small
OUTPUT_DIR=outputs
```

Use `.` when the adapter files are in the `MusicGen-AI` project folder. Otherwise, use the full path to the folder containing both adapter files. Relative paths are resolved from the project folder. Do not commit `.env`.

## Start the local app

From the `MusicGen-AI` folder:

```powershell
python app.py
```

Open **http://127.0.0.1:8000** in your browser. The server listens only on your own computer. Press `Ctrl+C` in the terminal to stop it.

The first track can take longer because MusicGen downloads its base model if it is not already cached. Generation runs on CUDA when available; CPU generation may be very slow. Generated WAV files are saved under `outputs/`.

## Run checks

```powershell
python -m unittest discover -s tests -v
```

These tests cover the local page, status endpoint, request validation, and generation API response. They do not run MusicGen inference.

## Troubleshooting

- **Adapter not found:** make sure the configured folder contains `adapter_config.json` and `adapter_model.safetensors` or `adapter_model.bin`.
- **Model or PEFT error:** the adapter must match the configured base model and the PEFT architecture used during training.
- **CUDA unavailable or out of memory:** verify the PyTorch build matches your GPU, lower the clip length, or close other GPU applications.
- **Model download error:** check internet access and confirm the model repository is reachable.

The app loads your saved adapter; it does not train it. A successful generation alone does not prove that fine-tuning was successful.
