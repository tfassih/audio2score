from __future__ import annotations

import numpy as np
import librosa

from .models import RhythmAnalysis


def _dedupe_beats(times: np.ndarray, min_gap: float) -> np.ndarray:
    if times.size == 0:
        return times
    out = [float(times[0])]
    for t in times[1:]:
        if float(t) - out[-1] >= min_gap:
            out.append(float(t))
    return np.asarray(out, dtype=float)


def _fill_missing_beats(times: np.ndarray, target_period: float) -> np.ndarray:
    if times.size < 2:
        return times
    out = [float(times[0])]
    for t in times[1:]:
        prev = out[-1]
        gap = float(t) - prev
        if gap > 1.55 * target_period:
            n = max(1, int(round(gap / target_period)))
            step = gap / n
            for k in range(1, n):
                out.append(prev + k * step)
        out.append(float(t))
    return np.asarray(out, dtype=float)


def _normalize_quarter_beats(
    detected: np.ndarray,
    onset_env: np.ndarray,
    beat_frames: np.ndarray,
    raw_tempo: float,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, float]:
    """Convert librosa's pulse train to a quarter-note beat map.

    Sparse piano is often tracked at eighth-note pulse. Rather than discarding
    the tracked beats and generating a metronomic grid (v0.6 behavior), v0.7
    preserves local timing and only changes the metrical level.
    """
    if detected.size == 0:
        tempo = raw_tempo if np.isfinite(raw_tempo) and raw_tempo > 0 else 120.0
        if tempo > 150:
            tempo *= 0.5
        elif tempo < 55:
            tempo *= 2.0
        return detected, float(tempo)

    if raw_tempo >= 135.0 and detected.size >= 4:
        # Choose even or odd eighth-note phase using onset strength at the
        # detected beat frames. This avoids arbitrarily shifting beat 1.
        even_idx = np.arange(0, len(beat_frames), 2)
        odd_idx = np.arange(1, len(beat_frames), 2)
        def phase_score(indices):
            if len(indices) == 0:
                return -1e9
            vals = []
            for i in indices:
                f = int(np.clip(beat_frames[i], 0, len(onset_env)-1))
                vals.append(float(onset_env[f]))
            # Slight preference for a phase that begins near the first attack.
            return float(np.mean(vals)) + (0.05 if indices[0] == 0 else 0.0)
        use = even_idx if phase_score(even_idx) >= phase_score(odd_idx) else odd_idx
        q = detected[use]
        tempo = raw_tempo * 0.5
    elif raw_tempo < 55.0 and detected.size >= 2:
        vals = [float(detected[0])]
        for a, b in zip(detected, detected[1:]):
            vals.append((float(a) + float(b)) * 0.5)
            vals.append(float(b))
        q = np.asarray(vals, dtype=float)
        tempo = raw_tempo * 2.0
    else:
        q = detected.copy()
        tempo = raw_tempo

    if q.size >= 3:
        diffs = np.diff(q)
        med = float(np.median(diffs[(diffs > 0.25) & (diffs < 1.5)])) if np.any((diffs > 0.25) & (diffs < 1.5)) else 60.0 / max(tempo, 1.0)
        q = _dedupe_beats(q, max(0.22, med * 0.52))
        q = _fill_missing_beats(q, med)

        # Reject extreme single-beat jitter while retaining actual rubato.
        if q.size >= 4:
            intervals = np.diff(q)
            med = float(np.median(intervals))
            lo, hi = med * 0.72, med * 1.36
            clipped = np.clip(intervals, lo, hi)
            # Blend tracked interval with robustly clipped interval rather than
            # replacing it, preserving local expressive timing.
            intervals = 0.72 * intervals + 0.28 * clipped
            q2 = [float(q[0])]
            for d in intervals:
                q2.append(q2[-1] + float(d))
            q = np.asarray(q2, dtype=float)

        med = float(np.median(np.diff(q)))
        tempo = 60.0 / med if med > 1e-6 else tempo

    return q, float(tempo)


def analyze_rhythm(y: np.ndarray, sr: int) -> RhythmAnalysis:
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop_length = 256 if sr <= 12000 else 512
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

    detected = librosa.frames_to_time(
        np.asarray(beat_frames, dtype=int), sr=sr, hop_length=hop_length
    )
    beat_times, tempo_bpm = _normalize_quarter_beats(
        detected, onset_env, np.asarray(beat_frames, dtype=int),
        raw_tempo, sr, hop_length,
    )

    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="time",
        backtrack=True,
        delta=0.10,
        wait=1,
    )

    # Align beat zero to the first real attack when it is close to the first
    # tracked beat. Unlike v0.6, do not force every later beat onto a constant
    # period: the variable beat map is the whole point.
    if beat_times.size:
        first_attack = float(onset_times[0]) if len(onset_times) else float(beat_times[0])
        if abs(float(beat_times[0]) - first_attack) <= 0.20:
            shift = first_attack - float(beat_times[0])
            beat_times = beat_times + shift

        # Extrapolate enough beats to cover the file.
        if beat_times.size >= 2:
            med = float(np.median(np.diff(beat_times)))
        else:
            med = 60.0 / max(tempo_bpm, 1.0)
        while beat_times.size and beat_times[-1] < duration + med:
            beat_times = np.append(beat_times, beat_times[-1] + med)
    else:
        period = max(60.0 / tempo_bpm, 0.20)
        beat_times = np.arange(0.0, duration + period * 1.5, period, dtype=float)

    # If the performance begins essentially immediately, expose a clean zero
    # for score mapping while retaining all subsequent local timing.
    if beat_times.size and beat_times[0] <= 0.25:
        beat_times = beat_times - beat_times[0]
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
