import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from music_engine.model import load_musicgen


def build_prompt(prompt: str, genre: str, instrumental: bool) -> str:
    parts = []
    if genre and genre != "No specific genre":
        parts.append(f"{genre} music")
    parts.append(prompt.strip())
    if instrumental and "instrumental" not in prompt.lower():
        parts.append("instrumental, no vocals")
    return ", ".join(parts)


def generate_music(
    prompt: str,
    genre: str,
    duration_seconds: int,
    instrumental: bool,
    output_dir: Path,
) -> dict:
    processor, model, device = load_musicgen()
    final_prompt = build_prompt(prompt, genre, instrumental)

    inputs = processor(text=[final_prompt], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # MusicGen uses approximately 50 generated audio tokens per second.
    # Longer clips require more GPU memory and generation time.
    max_new_tokens = max(1, int(duration_seconds * 50))

    with torch.inference_mode():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    audio = audio_values[0].detach().float().cpu().numpy()
    # MusicGen commonly returns (channels, samples). Convert to mono for a simple WAV output.
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    audio = np.asarray(audio, dtype=np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    sample_rate = int(processor.feature_extractor.sampling_rate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_genre = re.sub(r"[^a-zA-Z0-9_-]+", "_", genre.lower()).strip("_") or "music"
    filename = f"{safe_genre}_{int(time.time())}.wav"
    output_path = output_dir / filename
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")

    return {
        "path": str(output_path),
        "prompt": final_prompt,
        "duration": len(audio) / sample_rate,
        "sample_rate": sample_rate,
    }
