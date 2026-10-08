from __future__ import annotations

from collections import Counter
import math

import librosa
import numpy as np

from .models import (
    NoteEvent, ChordEvent, DynamicEvent, WedgeEvent, PedalEvent,
    PhraseSpan, SectionMarker, EngravingPlan,
)


def _sb(n: NoteEvent) -> float:
    return float(n.start_beat or 0.0)


def _eb(n: NoteEvent) -> float:
    return float(n.end_beat or 0.0)


def _parse_meter(meter: str) -> tuple[int, int, float]:
    beats, beat_type = meter.split("/", 1)
    b, bt = int(beats), int(beat_type)
    return b, bt, b * (4.0 / bt)


def _time_for_beat(beat: float, beat_times: list[float]) -> float:
    if not beat_times:
        return 0.0
    bt = np.asarray(beat_times, dtype=float)
    idx = np.arange(len(bt), dtype=float)
    if len(bt) == 1:
        return float(bt[0])
    if beat <= 0:
        return float(bt[0] + beat * (bt[1] - bt[0]))
    if beat >= len(bt) - 1:
        return float(bt[-1] + (beat - (len(bt) - 1)) * (bt[-1] - bt[-2]))
    return float(np.interp(beat, idx, bt))


def infer_pickup_beats(notes: list[NoteEvent], meter: str) -> float:
    """Conservative pickup detector.

    It only declares an anacrusis when low/bass attacks strongly agree on a
    recurring non-zero bar phase and notes actually exist before the inferred
    first downbeat. False pickups are much worse than missed pickups, so the
    threshold is intentionally high.
    """
    _, _, measure_beats = _parse_meter(meter)
    structural = [
        n for n in notes
        if n.hand == "left" and (n.role == "bass" or n.midi_pitch <= 52)
        and n.confidence >= 0.55
    ]
    if len(structural) < 8:
        return 0.0
    step = 0.25
    bins = Counter(round((_sb(n) % measure_beats) / step) * step for n in structural)
    phase, count = bins.most_common(1)[0]
    if phase < 0.375 or phase > measure_beats - 0.375:
        return 0.0
    if count / len(structural) < 0.64:
        return 0.0
    if not any(_sb(n) < phase - 0.125 for n in notes):
        return 0.0
    return float(round(phase / step) * step)


def detect_phrases(notes: list[NoteEvent], meter: str) -> list[PhraseSpan]:
    melody = sorted(
        [n for n in notes if n.hand == "right" and n.role == "melody"],
        key=lambda n: (_sb(n), n.midi_pitch),
    )
    if len(melody) < 2:
        return []
    _, _, measure_beats = _parse_meter(meter)
    phrases: list[list[NoteEvent]] = []
    current = [melody[0]]

    for n in melody[1:]:
        prev = current[-1]
        gap = _sb(n) - _eb(prev)
        phrase_len = _eb(prev) - _sb(current[0])
        at_bar = abs((_sb(n) / measure_beats) - round(_sb(n) / measure_beats)) < 0.05
        leap = abs(n.midi_pitch - prev.midi_pitch)
        should_break = (
            gap >= 0.75
            or phrase_len >= 10.0
            or (at_bar and phrase_len >= 4.0 and (gap >= 0.25 or leap >= 7))
        )
        if should_break and len(current) >= 2:
            phrases.append(current)
            current = [n]
        else:
            current.append(n)
    if len(current) >= 2:
        phrases.append(current)

    out = []
    for g in phrases:
        # Avoid slurring extremely short fragments or whole pages.
        start, end = _sb(g[0]), _eb(g[-1])
        if 0.75 <= end - start <= 12.0:
            out.append(PhraseSpan(start, end, g[0].midi_pitch, g[-1].midi_pitch))
    return out


def detect_dynamics(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
    max_beat: float,
    meter: str,
) -> tuple[list[DynamicEvent], list[WedgeEvent], list[float]]:
    _, _, measure_beats = _parse_meter(meter)
    n_measures = max(1, int(math.ceil(max_beat / measure_beats)))
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))

    measure_db: list[float] = []
    for m in range(n_measures):
        b0, b1 = m * measure_beats, min(max_beat, (m + 1) * measure_beats)
        t0, t1 = _time_for_beat(b0, beat_times), _time_for_beat(b1, beat_times)
        mask = (times >= t0) & (times < max(t1, t0 + 0.05))
        value = float(np.percentile(db[mask], 70)) if np.any(mask) else (measure_db[-1] if measure_db else -40.0)
        measure_db.append(value)

    if len(measure_db) >= 3:
        arr = np.asarray(measure_db)
        smoothed = np.copy(arr)
        for i in range(len(arr)):
            smoothed[i] = np.median(arr[max(0, i-1):min(len(arr), i+2)])
        measure_db = smoothed.tolist()

    lo = float(np.percentile(measure_db, 15))
    hi = float(np.percentile(measure_db, 88))
    span = max(4.0, hi - lo)
    marks = []
    for value in measure_db:
        z = (value - lo) / span
        marks.append("p" if z < 0.23 else "mp" if z < 0.48 else "mf" if z < 0.75 else "f")

    order = {"p": 0, "mp": 1, "mf": 2, "f": 3}
    dynamics: list[DynamicEvent] = []
    last = None
    last_measure = -99
    for m, mark in enumerate(marks):
        if m == 0 or (mark != last and m - last_measure >= 2):
            dynamics.append(DynamicEvent(m * measure_beats, mark))
            last, last_measure = mark, m

    wedges: list[WedgeEvent] = []
    for a, b in zip(dynamics, dynamics[1:]):
        da = order[a.mark]
        dbv = order[b.mark]
        if abs(dbv - da) >= 1 and b.beat - a.beat >= measure_beats * 2:
            wedges.append(WedgeEvent(
                a.beat + measure_beats * 0.5,
                b.beat - measure_beats * 0.25,
                "crescendo" if dbv > da else "diminuendo",
            ))
    return dynamics, wedges, measure_db


def detect_pedal(chords: list[ChordEvent], max_beat: float) -> list[PedalEvent]:
    """One sustain gesture per stable harmony region, capped at one measure.

    This is an engraving suggestion, not literal pedal-recognition. Releasing
    at harmony changes produces a useful default that pianists can edit.
    """
    out: list[PedalEvent] = []
    for c in chords:
        start = max(0.0, float(c.start_beat))
        end = min(float(c.end_beat), start + 4.0, max_beat)
        if end - start >= 0.75:
            out.append(PedalEvent(start, end))
    return out


def _section_fingerprint(
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    start: float,
    end: float,
    measure_beats: float,
) -> tuple[tuple, np.ndarray, float]:
    # Chord identity sampled once per measure.
    chord_seq = []
    b = start
    while b < end - 1e-9:
        c = next((x for x in chords if x.start_beat <= b + 0.05 < x.end_beat), None)
        chord_seq.append((c.root_pc, c.quality) if c else (-1, ""))
        b += measure_beats

    melody = [n for n in notes if n.role == "melody" and start <= _sb(n) < end]
    hist = np.zeros(12, dtype=float)
    for n in melody:
        hist[n.midi_pitch % 12] += max(0.25, min(2.0, _eb(n) - _sb(n)))
    norm = float(np.linalg.norm(hist))
    if norm > 0:
        hist /= norm
    density = len(melody) / max(1.0, (end - start))
    return tuple(chord_seq), hist, float(density)


def detect_sections(
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    max_beat: float,
    meter: str,
) -> list[SectionMarker]:
    _, _, measure_beats = _parse_meter(meter)
    section_len = measure_beats * 8.0
    if max_beat < section_len * 1.25:
        section_len = measure_beats * 4.0
    starts = list(np.arange(0.0, max_beat, section_len))
    if len(starts) <= 1:
        return [SectionMarker(0.0, "A", "A", 0)]

    prototypes: list[tuple[str, tuple, np.ndarray, float]] = []
    repeat_counts: Counter[str] = Counter()
    markers: list[SectionMarker] = []

    for start in starts:
        end = min(max_beat, start + section_len)
        if end - start < measure_beats * 2:
            continue
        chord_seq, hist, density = _section_fingerprint(notes, chords, start, end, measure_beats)

        best_family = None
        best_sim = -1.0
        for family, pchords, phist, pdensity in prototypes:
            n = max(len(chord_seq), len(pchords), 1)
            same = sum(1 for a, b in zip(chord_seq, pchords) if a == b) / n
            hsim = float(np.dot(hist, phist)) if np.linalg.norm(hist) and np.linalg.norm(phist) else 0.0
            dsim = 1.0 - min(1.0, abs(density - pdensity) / max(0.15, max(density, pdensity)))
            sim = 0.55 * same + 0.35 * hsim + 0.10 * dsim
            if sim > best_sim:
                best_sim, best_family = sim, family

        if best_family is None or best_sim < 0.72:
            family = chr(ord("A") + min(len(prototypes), 25))
            prototypes.append((family, chord_seq, hist.copy(), density))
            repeat_index = 0
        else:
            family = best_family
            repeat_counts[family] += 1
            repeat_index = repeat_counts[family]

        primes = "′" * min(repeat_index, 3)
        label = family + primes
        markers.append(SectionMarker(float(start), label, family, repeat_index))
    return markers


def build_engraving_plan(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    meter: str = "4/4",
) -> EngravingPlan:
    max_beat = max(
        [0.0]
        + [_eb(n) for n in notes]
        + [float(c.end_beat) for c in chords]
    )
    dynamics, wedges, _ = detect_dynamics(y, sr, beat_times, max_beat, meter)
    return EngravingPlan(
        dynamics=dynamics,
        wedges=wedges,
        pedals=detect_pedal(chords, max_beat),
        phrases=detect_phrases(notes, meter),
        sections=detect_sections(notes, chords, max_beat, meter),
        pickup_beats=infer_pickup_beats(notes, meter),
    )


def plan_for_variant(base: EngravingPlan | None, notes: list[NoteEvent], meter: str) -> EngravingPlan | None:
    """Reuse global performance marks but recompute slurs for each arrangement."""
    if base is None:
        return None
    return EngravingPlan(
        dynamics=list(base.dynamics),
        wedges=list(base.wedges),
        pedals=list(base.pedals),
        phrases=detect_phrases(notes, meter),
        sections=list(base.sections),
        pickup_beats=base.pickup_beats,
    )
