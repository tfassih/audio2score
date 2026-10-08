from __future__ import annotations

from pathlib import Path
import logging

import librosa
import numpy as np

from .models import NoteEvent
from .music import note_name_to_hz

log = logging.getLogger(__name__)


def _merge_notes(notes: list[NoteEvent], max_gap: float = 0.08) -> list[NoteEvent]:
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.start_sec, n.midi_pitch))
    out = [notes[0]]
    for n in notes[1:]:
        p = out[-1]
        if n.midi_pitch == p.midi_pitch and n.start_sec - p.end_sec <= max_gap:
            p.end_sec = max(p.end_sec, n.end_sec)
            p.confidence = max(p.confidence, n.confidence)
        else:
            out.append(n)
    return out


def transcribe_pyin(
    y: np.ndarray,
    sr: int,
    fmin: str = "C2",
    fmax: str = "C7",
    hop_length: int = 256,
    min_note_sec: float = 0.09,
    min_voiced_prob: float = 0.50,
) -> list[NoteEvent]:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=note_name_to_hz(fmin),
        fmax=note_name_to_hz(fmax),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    if f0 is None:
        return []
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    midi = np.full(len(f0), np.nan, dtype=float)
    valid = np.isfinite(f0) & voiced_flag & (voiced_prob >= min_voiced_prob)
    midi[valid] = np.rint(librosa.hz_to_midi(f0[valid]))

    # Median smoothing within voiced neighborhoods suppresses singer vibrato and frame jitter.
    smooth = midi.copy()
    for i in range(len(midi)):
        if not np.isfinite(midi[i]):
            continue
        lo, hi = max(0, i - 2), min(len(midi), i + 3)
        vals = midi[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            smooth[i] = float(np.median(vals))

    notes: list[NoteEvent] = []
    i = 0
    frame_dur = hop_length / sr
    while i < len(smooth):
        if not np.isfinite(smooth[i]):
            i += 1
            continue
        pitch = int(smooth[i])
        start = i
        j = i + 1
        gap = 0
        while j < len(smooth):
            if np.isfinite(smooth[j]) and int(smooth[j]) == pitch:
                gap = 0
                j += 1
                continue
            if not np.isfinite(smooth[j]) and gap < 1:
                gap += 1
                j += 1
                continue
            break
        end_idx = max(start + 1, j - gap)
        start_sec = float(times[start])
        end_sec = float(times[min(end_idx - 1, len(times) - 1)] + frame_dur)
        if end_sec - start_sec >= min_note_sec:
            probs = voiced_prob[start:end_idx]
            probs = probs[np.isfinite(probs)]
            conf = float(np.mean(probs)) if probs.size else 0.5
            notes.append(NoteEvent(start_sec, end_sec, pitch, conf, source="pyin"))
        i = max(j, i + 1)
    return _merge_notes(notes)


def reduce_to_monophonic(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Reduce a possibly polyphonic prediction to a plausible single lead line.

    The lead score favors confidence, moderate-to-high pitch, and continuity.
    This is intentionally conservative; source-separated vocals usually need little pruning.
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.start_sec, -n.confidence, -n.midi_pitch))
    out: list[NoteEvent] = []
    for n in notes:
        if n.end_sec <= n.start_sec:
            continue
        overlaps = [i for i, p in enumerate(out) if not (n.end_sec <= p.start_sec or n.start_sec >= p.end_sec)]
        if not overlaps:
            out.append(n)
            out.sort(key=lambda x: x.start_sec)
            continue
        # Compare against the most overlapping existing note.
        idx = overlaps[-1]
        p = out[idx]
        continuity = 0.0
        prev = out[idx - 1] if idx > 0 else None
        if prev:
            continuity = max(0.0, 1.0 - abs(n.midi_pitch - prev.midi_pitch) / 12.0)
        n_score = 0.72 * n.confidence + 0.20 * continuity + 0.08 * min(1.0, max(0.0, (n.midi_pitch - 48) / 36.0))
        p_cont = 0.0
        if prev:
            p_cont = max(0.0, 1.0 - abs(p.midi_pitch - prev.midi_pitch) / 12.0)
        p_score = 0.72 * p.confidence + 0.20 * p_cont + 0.08 * min(1.0, max(0.0, (p.midi_pitch - 48) / 36.0))
        if n_score > p_score + 0.05:
            out[idx] = n
    return _merge_notes(sorted(out, key=lambda x: x.start_sec))


def transcribe_melody(
    path: str | Path,
    y: np.ndarray,
    sr: int,
    backend: str = "auto",
    fmin: str = "C2",
    fmax: str = "C7",
) -> tuple[list[NoteEvent], str]:
    if backend not in {"auto", "pyin"}:
        raise ValueError(f"Unknown melody backend: {backend}")
    # Basic Pitch currently targets older Python releases, while Audio2Score
    # v0.6.0 deliberately targets Python 3.12+.  pYIN is therefore the
    # supported lead-melody backend in this environment.
    return transcribe_pyin(y, sr, fmin=fmin, fmax=fmax), "pyin"
