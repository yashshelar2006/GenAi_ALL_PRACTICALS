import os
from functools import lru_cache
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoProcessor, MusicgenForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL_ID = "facebook/musicgen-small"


def get_adapter_path() -> str:
    adapter_path = os.getenv("LORA_ADAPTER_PATH", "").strip()
    if adapter_path:
        adapter = Path(adapter_path).expanduser()
        if not adapter.is_absolute():
            adapter = PROJECT_ROOT / adapter
        return str(adapter.resolve())

    candidates = [
        PROJECT_ROOT,
        PROJECT_ROOT / "musicgen-small-lora-genres-final",
        Path.home() / "Downloads" / "musicgen-small-lora-genres-final",
    ]

    for candidate in candidates:
        if candidate.is_dir() and (
            (candidate / "adapter_config.json").exists()
            and ((candidate / "adapter_model.safetensors").exists() or (candidate / "adapter_model.bin").exists())
        ):
            return str(candidate)

    return ""


def musicgen_is_loaded() -> bool:
    return load_musicgen.cache_info().currsize > 0


@lru_cache(maxsize=1)
def load_musicgen():
    """Load the MusicGen base model and attach the trained PEFT LoRA adapter."""
    model_id = os.getenv("MUSICGEN_MODEL_ID", DEFAULT_MODEL_ID)
    adapter_path = get_adapter_path()

    if not adapter_path:
        raise ValueError(
            "LORA_ADAPTER_PATH is not set. Add the adapter folder path to your .env file, "
            "for example: LORA_ADAPTER_PATH=C:/Users/siddh/Downloads/musicgen-small-lora-genres-final"
        )

    adapter = Path(adapter_path).expanduser()
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter folder does not exist: {adapter}")

    config_path = adapter / "adapter_config.json"
    weights_present = (
        (adapter / "adapter_model.safetensors").exists()
        or (adapter / "adapter_model.bin").exists()
    )
    if not config_path.exists() or not weights_present:
        raise FileNotFoundError(
            f"Adapter folder must contain adapter_config.json and adapter_model.safetensors "
            f"(or adapter_model.bin): {adapter}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = MusicgenForConditionalGeneration.from_pretrained(model_id)

    # This assumes the adapter was trained for this base model and is a PEFT-compatible adapter.
    model = PeftModel.from_pretrained(base_model, str(adapter))
    model.to(device)
    model.eval()

    return processor, model, device
