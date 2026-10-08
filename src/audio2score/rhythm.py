from __future__ import annotations

import numpy as np
import librosa

from .models import RhythmAnalysis


def analyze_rhythm(y: np.ndarray, sr: int) -> RhythmAnalysis:
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        trim=False,
    )
    raw_tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 120.0
    if not np.isfinite(raw_tempo) or raw_tempo <= 0:
        raw_tempo = 120.0

    # Piano and sparse acoustic material is frequently tracked at eighth-note
    # pulse rather than quarter-note pulse. Normalize obvious double/half-time
    # estimates into a notation-friendly quarter-note range.
    tempo_bpm = raw_tempo
    if tempo_bpm > 150.0:
        tempo_bpm *= 0.5
    elif tempo_bpm < 55.0:
        tempo_bpm *= 2.0
    period = max(60.0 / tempo_bpm, 0.20)

    # If the recording starts with a clear attack, treat the file start as beat
    # zero. This greatly improves bar alignment for solo-piano recordings.
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="time",
        backtrack=True,
    )
    if len(onset_times) and float(onset_times[0]) <= 0.20:
        start = 0.0
    else:
        detected = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        if detected.size:
            phase = float(detected[0]) % period
            start = phase
            while start - period >= -0.15:
                start -= period
            start = max(0.0, start)
        else:
            start = 0.0

    beat_times = np.arange(start, duration + period * 1.5, period, dtype=float)
    if beat_times.size == 0 or beat_times[0] > 1e-6:
        beat_times = np.insert(beat_times, 0, 0.0)
    beat_times[0] = 0.0

    return RhythmAnalysis(
        tempo_bpm=float(tempo_bpm),
        beat_times=beat_times.tolist(),
        duration_sec=duration,
    )


def time_to_beat(time_sec: float, beat_times: list[float]) -> float:
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2:
        return 0.0
    period_first = bt[1] - bt[0]
    period_last = bt[-1] - bt[-2]
    if time_sec < bt[0]:
        return float((time_sec - bt[0]) / period_first)
    if time_sec > bt[-1]:
        return float((bt.size - 1) + (time_sec - bt[-1]) / period_last)
    return float(np.interp(time_sec, bt, np.arange(bt.size, dtype=float)))
