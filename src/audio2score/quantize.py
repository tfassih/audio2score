from __future__ import annotations

from copy import deepcopy
from collections import defaultdict

from .models import NoteEvent
from .rhythm import time_to_beat


def _quantize_one(note: NoteEvent, beat_times: list[float], step: float, min_duration_steps: int) -> NoteEvent:
    n = deepcopy(note)
    s = time_to_beat(n.start_sec, beat_times)
    e = time_to_beat(n.end_sec, beat_times)
    qs = round(s / step) * step
    qe = round(e / step) * step
    if qe <= qs:
        qe = qs + step * min_duration_steps
    n.start_beat = max(0.0, qs)
    n.end_beat = max(n.start_beat + step * min_duration_steps, qe)
    return n


def quantize_notes(
    notes: list[NoteEvent],
    beat_times: list[float],
    subdivisions_per_beat: int = 4,
    min_duration_steps: int = 1,
) -> list[NoteEvent]:
    """Quantize a monophonic melody (legacy/general-mode behavior)."""
    if subdivisions_per_beat < 1:
        raise ValueError("subdivisions_per_beat must be >= 1")
    step = 1.0 / subdivisions_per_beat
    out = [_quantize_one(n, beat_times, step, min_duration_steps) for n in notes]
    out.sort(key=lambda x: (x.start_beat or 0.0, x.midi_pitch))
    cleaned: list[NoteEvent] = []
    for n in out:
        if cleaned:
            p = cleaned[-1]
            ps, pe = p.start_beat or 0.0, p.end_beat or 0.0
            ns, ne = n.start_beat or 0.0, n.end_beat or 0.0
            if n.midi_pitch == p.midi_pitch and ns <= pe + step:
                p.end_beat = max(pe, ne)
                p.confidence = max(p.confidence, n.confidence)
                continue
            if ns < pe:
                p.end_beat = ns
                if (p.end_beat - ps) < step:
                    cleaned.pop()
        cleaned.append(n)
    return [n for n in cleaned if (n.end_beat or 0) > (n.start_beat or 0)]


def quantize_polyphonic_notes(
    notes: list[NoteEvent],
    beat_times: list[float],
    subdivisions_per_beat: int = 4,
    min_duration_steps: int = 1,
    max_notes_per_hand_attack: int = 5,
) -> list[NoteEvent]:
    """Quantize piano notes without destroying simultaneous voices."""
    if subdivisions_per_beat < 1:
        raise ValueError("subdivisions_per_beat must be >= 1")
    step = 1.0 / subdivisions_per_beat
    out = [_quantize_one(n, beat_times, step, min_duration_steps) for n in notes]

    # For the same physical key, a retrigger ends the prior keypress rather
    # than creating overlapping copies of one pitch.
    by_key: dict[tuple[str | None, int], list[NoteEvent]] = defaultdict(list)
    for n in out:
        by_key[(n.hand, n.midi_pitch)].append(n)
    cleaned: list[NoteEvent] = []
    for _, seq in by_key.items():
        seq.sort(key=lambda n: ((n.start_beat or 0.0), -n.confidence))
        kept: list[NoteEvent] = []
        for n in seq:
            if kept and abs((n.start_beat or 0.0) - (kept[-1].start_beat or 0.0)) < step * 0.45:
                if n.confidence > kept[-1].confidence:
                    kept[-1] = n
                continue
            if kept and (kept[-1].end_beat or 0.0) > (n.start_beat or 0.0):
                kept[-1].end_beat = n.start_beat
            kept.append(n)
        cleaned.extend(n for n in kept if (n.end_beat or 0.0) > (n.start_beat or 0.0))

    # Limit obviously over-detected attack clusters while preserving the
    # lowest LH and highest RH voices plus the most confident inner tones.
    clusters: dict[tuple[str | None, float], list[NoteEvent]] = defaultdict(list)
    for n in cleaned:
        clusters[(n.hand, float(n.start_beat or 0.0))].append(n)
    pruned: list[NoteEvent] = []
    for (hand, _), group in clusters.items():
        if len(group) <= max_notes_per_hand_attack:
            pruned.extend(group)
            continue
        extreme = min(group, key=lambda n: n.midi_pitch) if hand == "left" else max(group, key=lambda n: n.midi_pitch)
        ranked = sorted(group, key=lambda n: n.confidence, reverse=True)
        chosen = [extreme]
        for n in ranked:
            if n is extreme:
                continue
            if len(chosen) >= max_notes_per_hand_attack:
                break
            chosen.append(n)
        pruned.extend(chosen)

    return sorted(pruned, key=lambda n: ((n.start_beat or 0.0), 0 if n.hand == "left" else 1, n.midi_pitch))
