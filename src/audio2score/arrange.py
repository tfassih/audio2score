from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math

from .models import NoteEvent, ChordEvent
from .music import QUALITY_INTERVALS


def _sb(n: NoteEvent) -> float:
    return float(n.start_beat or 0.0)


def _eb(n: NoteEvent) -> float:
    return float(n.end_beat or 0.0)


def _chord_tones(c: ChordEvent) -> set[int]:
    return {(c.root_pc + i) % 12 for i in QUALITY_INTERVALS.get(c.quality, [0, 4, 7])}


def chord_at(chords: list[ChordEvent], beat: float) -> ChordEvent | None:
    for c in chords:
        if c.start_beat <= beat < c.end_beat:
            return c
    return chords[-1] if chords and beat >= chords[-1].start_beat else None


def infer_bass_inversions(chords: list[ChordEvent], notes: list[NoteEvent]) -> list[ChordEvent]:
    out = deepcopy(chords)
    lh = [n for n in notes if n.hand == "left" and _eb(n) > _sb(n)]
    for c in out:
        tones = _chord_tones(c)
        window_end = min(c.end_beat, c.start_beat + 1.25)
        candidates = [
            n for n in lh
            if c.start_beat - 0.125 <= _sb(n) < window_end
            and (n.midi_pitch % 12) in tones
            and n.midi_pitch <= 60
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda n: (n.midi_pitch, -n.confidence, -n.velocity, _sb(n)))
        floor = candidates[0].midi_pitch
        near_floor = [n for n in candidates if n.midi_pitch <= floor + 5]
        votes: Counter[int] = Counter()
        for n in near_floor:
            weight = max(1, int(round(4 * n.confidence + n.velocity / 32)))
            votes[n.midi_pitch % 12] += weight
        bass_pc, weight = votes.most_common(1)[0]
        total = sum(votes.values()) or 1
        if bass_pc in tones and weight / total >= 0.45:
            c.bass_pc = int(bass_pc)
    return out


def refine_hand_assignment(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Repair implausible cross-staff assignments with continuity memory."""
    out = deepcopy(notes)
    by_start: dict[float, list[NoteEvent]] = defaultdict(list)
    for n in out:
        by_start[round(_sb(n), 6)].append(n)

    prev_lh = 50.0
    prev_rh = 67.0
    for _, group in sorted(by_start.items()):
        group.sort(key=lambda n: n.midi_pitch)
        pitches = [n.midi_pitch for n in group]
        for n in group:
            if n.midi_pitch <= 51:
                n.hand = "left"
            elif n.midi_pitch >= 68:
                n.hand = "right"

        ambiguous = [n for n in group if 52 <= n.midi_pitch <= 67]
        for n in ambiguous:
            dl = abs(n.midi_pitch - prev_lh)
            dr = abs(n.midi_pitch - prev_rh)
            lower = [p for p in pitches if p < n.midi_pitch]
            upper = [p for p in pitches if p > n.midi_pitch]
            if lower and upper:
                below_gap = n.midi_pitch - max(lower)
                above_gap = min(upper) - n.midi_pitch
                if above_gap >= 6 and below_gap <= 4:
                    n.hand = "left"
                    continue
                if below_gap >= 6 and above_gap <= 4:
                    n.hand = "right"
                    continue
            n.hand = "left" if dl + 1.10 < dr else "right"

        lp = [n.midi_pitch for n in group if n.hand == "left"]
        rp = [n.midi_pitch for n in group if n.hand == "right"]
        if lp:
            prev_lh = 0.78 * prev_lh + 0.22 * (sum(lp) / len(lp))
        if rp:
            prev_rh = 0.78 * prev_rh + 0.22 * (sum(rp) / len(rp))
    return out


def _melody_groups(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    by_start: dict[float, list[NoteEvent]] = defaultdict(list)
    for n in notes:
        if n.hand == "right":
            by_start[round(_sb(n), 6)].append(n)
    return [sorted(g, key=lambda n: n.midi_pitch) for _, g in sorted(by_start.items()) if g]


def _track_melody_global(notes: list[NoteEvent]) -> set[int]:
    """Global upper-voice tracking via dynamic programming.

    Returns object ids of notes chosen as melody. Unlike the v0.4 highest-note
    heuristic, this penalizes implausible one-off octave/harmonic spikes and
    prefers a continuous singable line.
    """
    groups = _melody_groups(notes)
    if not groups:
        return set()

    scores: list[list[float]] = []
    back: list[list[int]] = []

    for gi, group in enumerate(groups):
        g_scores = [-1e9] * len(group)
        g_back = [-1] * len(group)
        top = max(n.midi_pitch for n in group)
        beat = _sb(group[0])
        if gi == 0:
            for j, n in enumerate(group):
                register = min(1.0, max(0.0, (n.midi_pitch - 57) / 24.0))
                top_bonus = 0.10 if n.midi_pitch == top else 0.0
                g_scores[j] = 1.45 * n.confidence + 0.16 * register + top_bonus + 0.08 * (n.velocity / 127.0)
        else:
            prev_group = groups[gi - 1]
            prev_scores = scores[-1]
            prev_beat = _sb(prev_group[0])
            gap = max(0.0, beat - prev_beat)
            for j, n in enumerate(group):
                emission = (
                    1.45 * n.confidence
                    + 0.08 * (n.velocity / 127.0)
                    + (0.10 if n.midi_pitch == top else 0.0)
                )
                best = -1e9
                best_k = -1
                for k, p in enumerate(prev_group):
                    leap = abs(n.midi_pitch - p.midi_pitch)
                    if gap >= 2.0:
                        leap_pen = 0.018 * max(0, leap - 7)
                    elif leap <= 2:
                        leap_pen = 0.00
                    elif leap <= 5:
                        leap_pen = 0.05
                    elif leap <= 7:
                        leap_pen = 0.12
                    elif leap <= 12:
                        leap_pen = 0.30
                    else:
                        leap_pen = 0.65 + 0.045 * (leap - 12)
                    # Weak high harmonics are especially suspect.
                    spike_pen = 0.0
                    if n.midi_pitch - p.midi_pitch >= 12 and n.confidence < 0.62:
                        spike_pen = 0.38
                    value = prev_scores[k] + emission - leap_pen - spike_pen
                    if value > best:
                        best, best_k = value, k
                g_scores[j] = best
                g_back[j] = best_k
        scores.append(g_scores)
        back.append(g_back)

    idx = max(range(len(groups[-1])), key=lambda j: scores[-1][j])
    chosen: list[NoteEvent] = []
    for gi in range(len(groups) - 1, -1, -1):
        chosen.append(groups[gi][idx])
        idx = back[gi][idx] if gi > 0 else -1
    return {id(n) for n in chosen}


def label_roles(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Label bass, melody and inner harmony using global melodic continuity."""
    out = deepcopy(notes)
    # deepcopy changes object identity, so run tracker on the copied collection.
    melody_ids = _track_melody_global(out)
    by_start: dict[float, list[NoteEvent]] = defaultdict(list)
    for n in out:
        by_start[round(_sb(n), 6)].append(n)

    for _, group in sorted(by_start.items()):
        for n in group:
            n.role = "harmony"
        lh = [n for n in group if n.hand == "left"]
        if lh:
            bass = min(lh, key=lambda n: (n.midi_pitch, -n.confidence))
            bass.role = "bass"
        for n in group:
            if id(n) in melody_ids:
                n.role = "melody"
                break
    return out


def label_roles_preserve_hands(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Label melody/bass/harmony without changing validated hand assignment.

    v0.8's faithful layer re-ran hand assignment and artifact pruning after
    validation. The user's v0.8 benchmark showed that the validated-performance
    MIDI was almost correct while the faithful representation reassigned 31
    notes between hands. v0.9 therefore treats validated hand labels and pitch
    hypotheses as authoritative for the faithful score.
    """
    out = deepcopy(notes)
    melody_ids = _track_melody_global(out)
    by_start: dict[float, list[NoteEvent]] = defaultdict(list)
    for n in out:
        if n.hand not in {"left", "right"}:
            # Only fill genuinely missing hand labels; never overwrite an
            # existing validator/backend decision.
            n.hand = "left" if n.midi_pitch < 60 else "right"
        by_start[round(_sb(n), 6)].append(n)

    for _, group in sorted(by_start.items()):
        for n in group:
            n.role = "harmony"
        lh = [n for n in group if n.hand == "left"]
        if lh:
            min(lh, key=lambda n: (n.midi_pitch, -n.confidence)).role = "bass"
        for n in group:
            if id(n) in melody_ids:
                n.role = "melody"
                break
    return out


def make_faithful_preserved(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Faithful notation source: preserve every validated pitch and hand.

    Quantization may alter score positions, but this function performs no
    additional pitch pruning, harmonic substitution, or hand reassignment.
    """
    return _dedupe_same_key(label_roles_preserve_hands(notes))


def prune_artifacts(notes: list[NoteEvent], chords: list[ChordEvent]) -> list[NoteEvent]:
    work = label_roles(refine_hand_assignment(notes))
    by_start: dict[float, list[NoteEvent]] = defaultdict(list)
    for n in work:
        by_start[round(_sb(n), 6)].append(n)

    kept: list[NoteEvent] = []
    harmonic_intervals = {12, 19, 24, 28, 31, 36}
    for beat, group in sorted(by_start.items()):
        c = chord_at(chords, beat)
        tones = _chord_tones(c) if c else set()
        group = sorted(group, key=lambda n: n.midi_pitch)
        for n in group:
            pc = n.midi_pitch % 12
            if n.role in {"melody", "bass"}:
                min_conf = 0.32
            elif pc in tones:
                min_conf = 0.40
            else:
                min_conf = 0.55
            if n.confidence < min_conf:
                continue

            alias = False
            if n.role == "harmony" and n.confidence < 0.74:
                for lower in group:
                    if lower.midi_pitch >= n.midi_pitch:
                        break
                    if n.midi_pitch - lower.midi_pitch in harmonic_intervals:
                        if lower.confidence >= n.confidence + 0.08 and (not tones or pc not in tones):
                            alias = True
                            break
            if not alias:
                kept.append(n)

    for n in kept:
        if n.role in {"melody", "bass"}:
            continue
        for c in chords:
            if _sb(n) + 0.25 < c.start_beat < _eb(n):
                if n.midi_pitch % 12 not in _chord_tones(c):
                    n.end_beat = c.start_beat
                    break

    kept = [n for n in kept if _eb(n) > _sb(n)]
    return label_roles(refine_hand_assignment(kept))


def _dedupe_same_key(notes: list[NoteEvent]) -> list[NoteEvent]:
    by: dict[tuple[str | None, int, float], NoteEvent] = {}
    for n in notes:
        k = (n.hand, n.midi_pitch, round(_sb(n), 6))
        old = by.get(k)
        if old is None or n.confidence > old.confidence:
            by[k] = n
    return sorted(by.values(), key=lambda n: (_sb(n), 0 if n.hand == "left" else 1, n.midi_pitch))


def make_faithful(notes: list[NoteEvent], chords: list[ChordEvent]) -> list[NoteEvent]:
    # v0.9: validated notes are the authority for the faithful layer.
    return make_faithful_preserved(notes)


def _nearest_pitch_for_pc(pc: int, target: int, lo: int, hi: int) -> int:
    options = [m for m in range(lo, hi + 1) if m % 12 == pc % 12]
    if not options:
        return target
    return min(options, key=lambda m: abs(m - target))


def _generated_intermediate_lh(chords: list[ChordEvent]) -> list[NoteEvent]:
    """Quarter-note broken-chord accompaniment: bass–5th–3rd–5th.

    It intentionally favors a stable keyboard pattern over trying to preserve
    every noisy detected inner voice.
    """
    out: list[NoteEvent] = []
    for c in chords:
        intervals = QUALITY_INTERVALS.get(c.quality, [0, 4, 7])
        third = intervals[1] if len(intervals) > 1 else 4
        fifth = intervals[2] if len(intervals) > 2 else 7
        bass_pc = c.bass_pc if c.bass_pc is not None else c.root_pc
        root_low = _nearest_pitch_for_pc(bass_pc, 42, 34, 48)
        third_mid = _nearest_pitch_for_pc((c.root_pc + third) % 12, 54, 48, 60)
        fifth_mid = _nearest_pitch_for_pc((c.root_pc + fifth) % 12, 55, 48, 62)

        bar = c.start_beat
        while bar < c.end_beat - 1e-9:
            bar_end = min(c.end_beat, bar + 4.0)
            pattern = [
                (0.0, root_low, "bass", 72),
                (1.0, fifth_mid, "accompaniment", 58),
                (2.0, third_mid, "accompaniment", 62),
                (3.0, fifth_mid, "accompaniment", 58),
            ]
            for off, pitch, role, vel in pattern:
                s = bar + off
                if s >= bar_end - 1e-9:
                    continue
                e = min(bar_end, s + 0.90)
                out.append(NoteEvent(
                    0.0, 0.0, pitch, confidence=1.0, source="arranged-intermediate",
                    start_beat=s, end_beat=e, velocity=vel, hand="left", role=role,
                ))
            bar += 4.0
    return out


def make_intermediate(faithful: list[NoteEvent], chords: list[ChordEvent]) -> list[NoteEvent]:
    work = deepcopy(faithful)
    by_start: dict[float, list[NoteEvent]] = defaultdict(list)
    for n in work:
        if n.hand != "right":
            continue
        n.start_beat = round(_sb(n) * 2) / 2
        n.end_beat = max(n.start_beat + 0.5, round(_eb(n) * 2) / 2)
        by_start[round(_sb(n), 6)].append(n)

    out: list[NoteEvent] = []
    for beat, group in sorted(by_start.items()):
        c = chord_at(chords, beat)
        tones = _chord_tones(c) if c else set()
        melody = [n for n in group if n.role == "melody"]
        chosen = melody[:1]
        candidates = sorted(
            [n for n in group if n not in chosen],
            key=lambda n: (
                (n.midi_pitch % 12) in tones,
                n.confidence,
                -abs(n.midi_pitch - (chosen[0].midi_pitch if chosen else 67)),
            ),
            reverse=True,
        )
        # At most melody + one inner tone keeps the RH readable.
        for n in candidates:
            if len(chosen) >= 2:
                break
            if not chosen or abs(n.midi_pitch - chosen[0].midi_pitch) <= 10:
                n.role = "harmony"
                chosen.append(n)
        out.extend(chosen)

    out.extend(_generated_intermediate_lh(chords))
    # Preserve generated accompaniment roles after hand/role cleanup by only
    # relabeling the right-hand melodic material.
    right = label_roles([n for n in out if n.hand == "right"])
    left = [n for n in out if n.hand == "left"]
    return _dedupe_same_key(left + right)


def _make_monophonic_melody(faithful: list[NoteEvent]) -> list[NoteEvent]:
    melody = sorted(
        [deepcopy(n) for n in faithful if n.hand == "right" and n.role == "melody"],
        key=lambda n: (_sb(n), n.midi_pitch),
    )
    seq: list[NoteEvent] = []
    for m in melody:
        m.start_beat = round(_sb(m) * 2) / 2
        m.end_beat = max(m.start_beat + 0.5, round(_eb(m) * 2) / 2)
        m.hand = "right"
        m.role = "melody"
        if seq and m.start_beat < _eb(seq[-1]):
            seq[-1].end_beat = m.start_beat
            if _eb(seq[-1]) <= _sb(seq[-1]):
                seq.pop()
        seq.append(m)
    return [n for n in seq if _eb(n) > _sb(n)]


def make_easy(faithful: list[NoteEvent], chords: list[ChordEvent]) -> list[NoteEvent]:
    """Melody with a simple half-note bass/fifth accompaniment."""
    out = _make_monophonic_melody(faithful)
    for c in chords:
        bass_pc = c.bass_pc if c.bass_pc is not None else c.root_pc
        intervals = QUALITY_INTERVALS.get(c.quality, [0, 4, 7])
        fifth_int = intervals[2] if len(intervals) > 2 else 7
        bass = _nearest_pitch_for_pc(bass_pc, 41, 34, 47)
        fifth = _nearest_pitch_for_pc((c.root_pc + fifth_int) % 12, 50, 45, 57)

        bar = c.start_beat
        while bar < c.end_beat - 1e-9:
            bar_end = min(c.end_beat, bar + 4.0)
            first_end = min(bar_end, bar + 2.0)
            out.append(NoteEvent(
                0.0, 0.0, bass, confidence=1.0, source="arranged-easy",
                start_beat=bar, end_beat=first_end, velocity=66, hand="left", role="bass",
            ))
            if bar + 2.0 < bar_end - 1e-9:
                out.append(NoteEvent(
                    0.0, 0.0, fifth, confidence=1.0, source="arranged-easy",
                    start_beat=bar + 2.0, end_beat=bar_end, velocity=56,
                    hand="left", role="accompaniment",
                ))
            bar += 4.0
    return _dedupe_same_key(out)


@dataclass
class ArrangementSet:
    faithful: list[NoteEvent]
    intermediate: list[NoteEvent]
    easy: list[NoteEvent]
    chords: list[ChordEvent]


def build_arrangements(notes: list[NoteEvent], chords: list[ChordEvent]) -> ArrangementSet:
    # Refine chord inversions using a copy if desired, but do not let that
    # process rewrite validated hand assignment in the faithful score.
    inferred_chords = infer_bass_inversions(chords, notes)
    faithful = make_faithful(notes, inferred_chords)
    intermediate = make_intermediate(faithful, inferred_chords)
    easy = make_easy(faithful, inferred_chords)
    return ArrangementSet(
        faithful=faithful,
        intermediate=intermediate,
        easy=easy,
        chords=inferred_chords,
    )
