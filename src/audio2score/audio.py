from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import librosa
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    vocals: Path | None = None
    instrumental: Path | None = None


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file contains no samples: {path}")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / max(1.0, peak)
    return y.astype(np.float32), actual_sr


def separate_vocals(
    path: str | Path,
    output_dir: str | Path,
    preset: str = "vocal_balanced",
) -> SeparationResult:
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "Vocal separation requested, but audio-separator is not installed. "
            "Install the optional separation dependencies first."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        output_dir=str(output_dir),
        output_format="WAV",
        ensemble_preset=preset,
    )
    separator.load_model()
    outputs = separator.separate(
        str(path),
        {"Vocals": "vocals", "Instrumental": "instrumental"},
    )

    result = SeparationResult()
    for item in outputs:
        p = Path(item)
        if not p.is_absolute():
            p = output_dir / p.name
        low = p.name.lower()
        if "vocal" in low:
            result.vocals = p
        elif "instrument" in low:
            result.instrumental = p

    # Current API allows custom names, but keep a filesystem fallback for future changes.
    if result.vocals is None:
        candidates = list(output_dir.glob("*vocal*.wav"))
        if candidates:
            result.vocals = candidates[0]
    if result.instrumental is None:
        candidates = list(output_dir.glob("*instrument*.wav"))
        if candidates:
            result.instrumental = candidates[0]

    if result.vocals is None:
        raise RuntimeError("Source separation completed, but no vocal stem could be identified.")
    log.info("Vocal stem: %s", result.vocals)
    if result.instrumental:
        log.info("Instrumental stem: %s", result.instrumental)
    return result
