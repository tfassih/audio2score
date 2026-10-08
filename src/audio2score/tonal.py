from __future__ import annotations

import numpy as np
import librosa

from .models import KeyEstimate, ChordEvent
from .music import QUALITY_INTERVALS, normalize

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)


def chroma_features(y: np.ndarray, sr: int, hop_length: int = 512, fast: bool = False) -> np.ndarray:
    if fast:
        return librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=4096,
            hop_length=hop_length,
        )
    y_harm = librosa.effects.harmonic(y)
    return librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)


def _key_candidates(chroma: np.ndarray) -> list[tuple[float, int, str]]:
    profile = np.nan_to_num(chroma, nan=0.0).mean(axis=1)
    if float(profile.sum()) <= 1e-9:
        return [(0.0, 0, "major")]
    profile = (profile - profile.mean()) / (profile.std() + 1e-8)
    candidates = []
    for tonic in range(12):
        for mode, ref in (("major", _MAJOR), ("minor", _MINOR)):
            rotated = np.roll(ref, tonic)
            rotated = (rotated - rotated.mean()) / (rotated.std() + 1e-8)
            score = float(np.dot(profile, rotated) / len(profile))
            candidates.append((score, tonic, mode))
    candidates.sort(reverse=True)
    return candidates


def estimate_key(chroma: np.ndarray, first_chord: ChordEvent | None = None) -> KeyEstimate:
    candidates = _key_candidates(chroma)
    best = candidates[0]
    chosen = best

    # Relative major/minor is a classic ambiguity in global pitch histograms.
    # If the piece opens clearly on the relative minor tonic, prefer that minor
    # key when its correlation is reasonably close to the relative major.
    if first_chord is not None and best[2] == "major":
        rel_minor = (best[1] + 9) % 12
        rel = next((c for c in candidates if c[1] == rel_minor and c[2] == "minor"), None)
        if (
            rel is not None
            and first_chord.root_pc == rel_minor
            and first_chord.quality == "min"
            and best[0] - rel[0] <= 0.18
        ):
            chosen = rel

    second = next((c for c in candidates if c != chosen), candidates[1] if len(candidates) > 1 else chosen)
    confidence = max(0.0, min(1.0, (chosen[0] - second[0] + 0.18) / 0.45))
    return KeyEstimate(chosen[1], chosen[2], confidence)


def _chord_states() -> list[tuple[int, str, np.ndarray, float]]:
    qualities = ("maj", "min", "dim", "7", "maj7", "min7")
    prior = {"maj": 0.035, "min": 0.035, "dim": -0.020, "7": -0.010, "maj7": -0.020, "min7": -0.020}
    states = []
    for root in range(12):
        for quality in qualities:
            intervals = QUALITY_INTERVALS[quality]
            template = np.zeros(12, dtype=float)
            for iv in intervals:
                template[(root + iv) % 12] = 1.0
            template[root] += 0.22
            states.append((root, quality, normalize(template), prior[quality]))
    return states


def _diatonic_prior(root: int, quality: str, key: KeyEstimate | None) -> float:
    if key is None:
        return 0.0
    rel = (root - key.tonic_pc) % 12
    if key.mode == "major":
        diatonic = {
            (0, "maj"), (2, "min"), (4, "min"), (5, "maj"),
            (7, "maj"), (9, "min"), (11, "dim"),
        }
    else:
        diatonic = {
            (0, "min"), (2, "dim"), (3, "maj"), (5, "min"),
            (7, "min"), (8, "maj"), (10, "maj"),
            # Harmonic-minor dominant is common enough to allow.
            (7, "maj"), (7, "7"),
        }
    if (rel, quality) in diatonic:
        return 0.075
    # Sevenths built on a diatonic triad are plausible, but do not outrank a
    # clear triad merely because melody/pedal adds a color tone.
    if quality in {"7", "maj7", "min7"}:
        triad_q = {"7": "maj", "maj7": "maj", "min7": "min"}[quality]
        if (rel, triad_q) in diatonic:
            return 0.025
    return -0.015


def detect_chords_barwise(
    chroma: np.ndarray,
    sr: int,
    beat_times: list[float],
    beats_per_bar: int = 4,
    hop_length: int = 512,
    key: KeyEstimate | None = None,
) -> list[ChordEvent]:
    """Detect one readable harmony label per measure.

    Barwise piano harmony deliberately uses triads only. Seventh/color tones
    from the melody and sustain pedal otherwise tend to turn a clear B-major
    measure into D#m7, for example. Richer chord labels can be added later
    after bass/voice separation is available.
    """
    if len(beat_times) < 2:
        return []

    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    states: list[tuple[int, str, np.ndarray, float]] = []
    for root in range(12):
        for quality, prior in (("maj", 0.020), ("min", 0.020), ("dim", -0.025)):
            template = np.zeros(12, dtype=float)
            for iv in QUALITY_INTERVALS[quality]:
                template[(root + iv) % 12] = 1.0
            # Do not root-boost the template here: arpeggiated piano voicings
            # often put a chord third in the foreground.
            states.append((root, quality, normalize(template), prior))

    n_beats = max(0, len(beat_times) - 1)
    n_bars = int(np.ceil(n_beats / beats_per_bar))
    labels = []

    for bar in range(n_bars):
        sb = bar * beats_per_bar
        eb = min((bar + 1) * beats_per_bar, n_beats)
        if eb <= sb:
            continue
        start = beat_times[sb]
        end = beat_times[eb]
        mask = (frame_times >= start) & (frame_times < end)
        vec = np.nanmean(chroma[:, mask], axis=1) if np.any(mask) else np.zeros(12)
        vec = normalize(np.maximum(vec, 0.0))

        scores = []
        for idx, (root, quality, template, base_prior) in enumerate(states):
            chord_mask = template > 0
            non_chord = float(vec[~chord_mask].sum())
            score = (
                float(np.dot(vec, template))
                - 0.035 * non_chord
                + base_prior
                + _diatonic_prior(root, quality, key)
            )
            scores.append((score, idx))

        scores.sort(reverse=True)
        best_score, best_idx = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else best_score
        root, quality, _, _ = states[best_idx]
        conf = float(np.clip((best_score - second_score + 0.025) / 0.14, 0.0, 1.0))
        labels.append((sb, eb, root, quality, conf))

    # Only smooth a clearly weak isolated label when both neighbors agree.
    smoothed = labels[:]
    for i in range(1, len(labels) - 1):
        prev = labels[i - 1]
        cur = labels[i]
        nxt = labels[i + 1]
        if (prev[2], prev[3]) == (nxt[2], nxt[3]) and (cur[2], cur[3]) != (prev[2], prev[3]):
            if cur[4] < 0.32:
                smoothed[i] = (
                    cur[0], cur[1], prev[2], prev[3],
                    min(prev[4], nxt[4]),
                )

    events: list[ChordEvent] = []
    i = 0
    while i < len(smoothed):
        sb, eb, root, quality, conf = smoothed[i]
        j = i + 1
        confs = [conf]
        final_eb = eb
        while j < len(smoothed) and (smoothed[j][2], smoothed[j][3]) == (root, quality):
            final_eb = smoothed[j][1]
            confs.append(smoothed[j][4])
            j += 1
        events.append(
            ChordEvent(
                start_beat=float(sb),
                end_beat=float(final_eb),
                root_pc=int(root),
                quality=str(quality),
                confidence=float(np.mean(confs)),
            )
        )
        i = j
    return events

def detect_chords(
    chroma: np.ndarray,
    sr: int,
    beat_times: list[float],
    hop_length: int = 512,
    change_penalty: float = 0.12,
) -> list[ChordEvent]:
    # Retained for full-mix v0.1 compatibility: beatwise chord detection.
    if len(beat_times) < 2:
        return []
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    states = _chord_states()
    n_beats = len(beat_times) - 1
    emissions = np.zeros((n_beats, len(states)), dtype=float)

    for b in range(n_beats):
        start, end = beat_times[b], beat_times[b + 1]
        mask = (frame_times >= start) & (frame_times < end)
        vec = np.nanmean(chroma[:, mask], axis=1) if np.any(mask) else np.zeros(12)
        vec = normalize(np.maximum(vec, 0.0))
        for s, (_, _, template, prior) in enumerate(states):
            emissions[b, s] = float(np.dot(vec, template) + prior)

    dp = np.full_like(emissions, -np.inf)
    back = np.zeros_like(emissions, dtype=np.int32)
    dp[0] = emissions[0]
    for b in range(1, n_beats):
        prev_best_idx = int(np.argmax(dp[b - 1]))
        prev_best = float(dp[b - 1, prev_best_idx])
        for s in range(len(states)):
            stay = float(dp[b - 1, s])
            change = prev_best - change_penalty
            if stay >= change:
                dp[b, s] = emissions[b, s] + stay
                back[b, s] = s
            else:
                dp[b, s] = emissions[b, s] + change
                back[b, s] = prev_best_idx

    path = np.zeros(n_beats, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for b in range(n_beats - 2, -1, -1):
        path[b] = back[b + 1, path[b + 1]]

    events = []
    i = 0
    while i < n_beats:
        root, quality, _, _ = states[int(path[i])]
        j = i + 1
        while j < n_beats and path[j] == path[i]:
            j += 1
        events.append(ChordEvent(float(i), float(j), root, quality, 0.5))
        i = j
    return events
