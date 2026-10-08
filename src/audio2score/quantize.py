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


def _adaptive_grid_steps(max_subdivisions: int) -> list[tuple[float, float]]:
    """Return (step, complexity_penalty) candidates for score notation.

    The penalties deliberately prefer simpler notation when two choices explain
    the performed timing almost equally well.
    """
    steps: list[tuple[float, float]] = [(1.0, 0.00), (0.5, 0.018)]
    # v0.6 MusicXML emits binary note values only. Tuplet engraving is kept
    # out of the quantizer until the exporter can write explicit time-modification
    # and tuplet brackets rather than rounding thirds onto a binary tick grid.
    if max_subdivisions >= 4:
        steps.append((0.25, 0.045))
    if max_subdivisions >= 8:
        steps.append((0.125, 0.085))
    return steps


def _candidate_beats(raw: float, max_subdivisions: int) -> list[tuple[float, float]]:
    cands: dict[float, float] = {}
    for step, complexity in _adaptive_grid_steps(max_subdivisions):
        center = round(raw / step)
        for k in (center - 1, center, center + 1):
            q = max(0.0, k * step)
            # Prefer strong metrical locations while preserving syncopation when
            # the performance timing clearly supports it.
            phase = q % 1.0
            strength_bonus = -0.035 if abs(phase) < 1e-7 else -0.012 if abs(phase - 0.5) < 1e-7 else 0.0
            penalty = complexity + strength_bonus
            old = cands.get(round(q, 8))
            if old is None or penalty < old:
                cands[round(q, 8)] = penalty
    return sorted((q, p) for q, p in cands.items())


def _attack_groups_seconds(notes: list[NoteEvent], tolerance_sec: float = 0.060) -> list[list[NoteEvent]]:
    seq = sorted(notes, key=lambda n: (n.start_sec, n.midi_pitch))
    groups: list[list[NoteEvent]] = []
    for n in seq:
        if not groups or n.start_sec - groups[-1][0].start_sec > tolerance_sec:
            groups.append([n])
        else:
            groups[-1].append(n)
    return groups


def adaptive_quantize_polyphonic_notes(
    notes: list[NoteEvent],
    beat_times: list[float],
    *,
    max_subdivisions: int = 4,
    min_duration_beats: float | None = None,
    max_notes_per_hand_attack: int = 5,
) -> list[NoteEvent]:
    """Context-aware piano quantization used only for score generation.

    Unlike the legacy fixed-grid function, this preserves the unquantized
    performance representation and chooses among quarter/eighth/triplet/
    sixteenth positions with a dynamic-programming cost that balances timing
    error, notation complexity, metrical strength, and local interval shape.
    """
    if max_subdivisions < 1:
        raise ValueError("max_subdivisions must be >= 1")
    if not notes:
        return []
    if len(beat_times) < 2:
        return quantize_polyphonic_notes(notes, beat_times, subdivisions_per_beat=max_subdivisions)

    work = [deepcopy(n) for n in notes]
    groups = _attack_groups_seconds(work)
    raw_beats = [
        time_to_beat(float(sum(n.start_sec for n in g) / len(g)), beat_times)
        for g in groups
    ]
    cand_lists = [_candidate_beats(rb, max_subdivisions) for rb in raw_beats]

    # Dynamic programming over attack locations.
    dp: list[list[float]] = []
    back: list[list[int]] = []
    for i, (raw, cands) in enumerate(zip(raw_beats, cand_lists)):
        row = [float("inf")] * len(cands)
        brow = [-1] * len(cands)
        for j, (q, complexity) in enumerate(cands):
            local = 2.6 * abs(q - raw) + complexity
            if i == 0:
                row[j] = local
                continue
            raw_interval = max(0.0, raw - raw_beats[i - 1])
            best = float("inf")
            best_k = -1
            for k, (pq, _) in enumerate(cand_lists[i - 1]):
                if q < pq - 1e-7:
                    continue
                q_interval = q - pq
                # Do not collapse clearly separate attacks onto one grid point.
                if raw_interval > 0.11 and q_interval < 1e-7:
                    transition = 1.2
                else:
                    transition = 0.42 * abs(q_interval - raw_interval)
                # Penalize implausibly tiny notated intervals unless the raw
                # performance also contains a fast ornament/repetition.
                if 0 < q_interval < 0.24 and raw_interval > 0.30:
                    transition += 0.35
                value = dp[i - 1][k] + local + transition
                if value < best:
                    best, best_k = value, k
            row[j] = best
            brow[j] = best_k
        dp.append(row)
        back.append(brow)

    idx = min(range(len(dp[-1])), key=lambda j: dp[-1][j])
    chosen = [0.0] * len(groups)
    for i in range(len(groups) - 1, -1, -1):
        chosen[i] = cand_lists[i][idx][0]
        idx = back[i][idx] if i > 0 else -1

    for group, q in zip(groups, chosen):
        for n in group:
            n.start_beat = q

    # Quantize keypress duration independently from acoustic sustain. Common
    # note values get lower complexity cost; unusual durations are allowed when
    # they explain the performed release materially better.
    duration_options = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    if max_subdivisions < 4:
        duration_options = [d for d in duration_options if d >= 0.5]
    duration_penalty = {
        0.25: 0.05, 0.5: 0.02,
        0.75: 0.05, 1.0: 0.0, 1.5: 0.03, 2.0: 0.0,
        3.0: 0.03, 4.0: 0.0, 6.0: 0.04, 8.0: 0.02,
    }
    min_duration = min_duration_beats if min_duration_beats is not None else 1.0 / max(max_subdivisions, 4)
    for n in work:
        raw_end = time_to_beat(n.end_sec, beat_times)
        raw_start = time_to_beat(n.start_sec, beat_times)
        raw_dur = max(min_duration, raw_end - raw_start)
        options = [d for d in duration_options if d >= min_duration]
        best_d = min(
            options,
            key=lambda d: 1.35 * abs(d - raw_dur) + duration_penalty.get(d, 0.08),
        )
        n.end_beat = float(n.start_beat or 0.0) + best_d

    # Same-key retrigger ends the previous notated note.
    by_key: dict[tuple[str | None, int], list[NoteEvent]] = defaultdict(list)
    for n in work:
        by_key[(n.hand, n.midi_pitch)].append(n)
    for seq in by_key.values():
        seq.sort(key=lambda n: (float(n.start_beat or 0.0), -n.confidence))
        for a, b in zip(seq, seq[1:]):
            if float(a.end_beat or 0.0) > float(b.start_beat or 0.0):
                a.end_beat = b.start_beat

    # Limit implausibly dense same-hand attacks, preserving the staff extreme
    # plus the most audio-supported/confident inner notes.
    clusters: dict[tuple[str | None, float], list[NoteEvent]] = defaultdict(list)
    for n in work:
        if float(n.end_beat or 0.0) <= float(n.start_beat or 0.0):
            continue
        clusters[(n.hand, round(float(n.start_beat or 0.0), 8))].append(n)
    out: list[NoteEvent] = []
    for (hand, _), group in clusters.items():
        if len(group) <= max_notes_per_hand_attack:
            out.extend(group)
            continue
        extreme = min(group, key=lambda n: n.midi_pitch) if hand == "left" else max(group, key=lambda n: n.midi_pitch)
        ranked = sorted(
            group,
            key=lambda n: (
                n.audio_support if n.audio_support is not None else n.confidence,
                n.confidence,
            ),
            reverse=True,
        )
        chosen_group = [extreme]
        for n in ranked:
            if n is extreme:
                continue
            if len(chosen_group) >= max_notes_per_hand_attack:
                break
            chosen_group.append(n)
        out.extend(chosen_group)

    return sorted(out, key=lambda n: (float(n.start_beat or 0.0), 0 if n.hand == "left" else 1, n.midi_pitch))
