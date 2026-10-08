from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from .models import NoteEvent, ChordEvent, KeyEstimate
from .music import QUALITY_INTERVALS
from .rhythm import time_to_beat
from .validation import (
    _pitch_matrix,
    _pitch_evidence,
    _onset_times,
)


_TRIADS = (
    ("maj", (0, 4, 7)),
    ("min", (0, 3, 7)),
    ("dim", (0, 3, 6)),
)


def _scale_prior(root: int, quality: str, key: KeyEstimate | None) -> float:
    if key is None:
        return 0.0
    rel = (root - key.tonic_pc) % 12
    if key.mode == "major":
        allowed = {
            (0, "maj"), (2, "min"), (4, "min"), (5, "maj"),
            (7, "maj"), (9, "min"), (11, "dim"),
        }
    else:
        allowed = {
            (0, "min"), (2, "dim"), (3, "maj"), (5, "min"),
            (7, "min"), (8, "maj"), (10, "maj"),
            (7, "maj"),
        }
    return 0.055 if (rel, quality) in allowed else -0.018


def _note_overlap_beats(
    n: NoteEvent,
    start_beat: float,
    end_beat: float,
    beat_times: list[float],
) -> float:
    ns = time_to_beat(n.start_sec, beat_times)
    ne = time_to_beat(n.end_sec, beat_times)
    return max(0.0, min(end_beat, ne) - max(start_beat, ns))


def refine_harmony_from_piano_notes(
    notes: list[NoteEvent],
    beat_times: list[float],
    key: KeyEstimate | None,
    *,
    beats_per_bar: int = 4,
    fallback: list[ChordEvent] | None = None,
) -> list[ChordEvent]:
    """Infer readable bar harmony from the validated piano transcription.

    v0.7 relied on chroma alone. Solo piano gives us a stronger source after
    validation: actual note hypotheses, especially the bass. Left-hand notes
    therefore receive substantially more weight than right-hand melody tones.
    The result remains a *soft* harmonic model used for cleanup/engraving.
    """
    if len(beat_times) < 2 or not notes:
        return deepcopy(fallback or [])

    max_beat = max(
        time_to_beat(max(n.end_sec, n.start_sec), beat_times)
        for n in notes
    )
    n_bars = max(1, int(math.ceil(max_beat / beats_per_bar)))
    bars: list[tuple[float, float, int, str, float, int | None]] = []

    for bar in range(n_bars):
        sb = float(bar * beats_per_bar)
        eb = sb + float(beats_per_bar)
        pc_energy = np.zeros(12, dtype=float)
        lh_starts: list[NoteEvent] = []

        for n in notes:
            overlap = _note_overlap_beats(n, sb, eb, beat_times)
            if overlap <= 0:
                continue
            support = (
                float(n.audio_support)
                if n.audio_support is not None
                else float(n.confidence)
            )
            hand_w = 2.25 if n.hand == "left" else 0.55
            dur_w = min(overlap, 2.0)
            vel_w = 0.82 + 0.36 * (max(1, min(127, n.velocity)) / 127.0)
            pc_energy[n.midi_pitch % 12] += (
                hand_w * dur_w * vel_w * (0.45 + 0.55 * support)
            )
            nb = time_to_beat(n.start_sec, beat_times)
            if n.hand == "left" and sb - 0.12 <= nb < min(eb, sb + 1.35):
                lh_starts.append(n)

        if float(pc_energy.sum()) <= 1e-9:
            continue
        vec = pc_energy / (float(np.linalg.norm(pc_energy)) + 1e-9)

        scores: list[tuple[float, int, str]] = []
        for root in range(12):
            for quality, ivs in _TRIADS:
                templ = np.zeros(12, dtype=float)
                for iv in ivs:
                    templ[(root + iv) % 12] = 1.0
                templ /= float(np.linalg.norm(templ)) + 1e-9
                score = float(np.dot(vec, templ))
                score += _scale_prior(root, quality, key)

                if lh_starts:
                    low = min(
                        lh_starts,
                        key=lambda n: (
                            n.midi_pitch,
                            -(n.audio_support if n.audio_support is not None else n.confidence),
                        ),
                    )
                    pcs = {(root + iv) % 12 for iv in ivs}
                    if low.midi_pitch % 12 == root:
                        score += 0.085
                    elif low.midi_pitch % 12 in pcs:
                        score += 0.035

                scores.append((score, root, quality))

        scores.sort(reverse=True)
        best_score, root, quality = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else best_score
        confidence = float(
            np.clip((best_score - second_score + 0.035) / 0.16, 0.0, 1.0)
        )

        # If the note evidence is genuinely indecisive, keep the existing
        # audio-derived bar harmony rather than forcing a new answer.
        if confidence < 0.18 and fallback:
            fb = next(
                (
                    c for c in fallback
                    if c.start_beat <= sb < c.end_beat
                ),
                None,
            )
            if fb is not None:
                root, quality = fb.root_pc, fb.quality

        tones = {
            (root + iv) % 12
            for iv in QUALITY_INTERVALS.get(quality, [0, 4, 7])
        }
        bass_pc = None
        candidates = [
            n for n in lh_starts
            if (n.midi_pitch % 12) in tones
        ]
        if candidates:
            low = min(candidates, key=lambda n: n.midi_pitch)
            bass_pc = int(low.midi_pitch % 12)

        bars.append((sb, eb, int(root), str(quality), confidence, bass_pc))

    if not bars:
        return deepcopy(fallback or [])

    out: list[ChordEvent] = []
    for sb, eb, root, quality, conf, bass_pc in bars:
        if (
            out
            and out[-1].root_pc == root
            and out[-1].quality == quality
            and out[-1].bass_pc == bass_pc
            and abs(out[-1].end_beat - sb) < 1e-7
        ):
            out[-1].end_beat = eb
            out[-1].confidence = float((out[-1].confidence + conf) / 2.0)
        else:
            out.append(
                ChordEvent(
                    start_beat=sb,
                    end_beat=eb,
                    root_pc=root,
                    quality=quality,
                    confidence=conf,
                    bass_pc=bass_pc,
                )
            )
    return out


def _chord_at(chords: list[ChordEvent], beat: float) -> ChordEvent | None:
    for c in chords:
        if c.start_beat <= beat < c.end_beat:
            return c
    return chords[-1] if chords and beat >= chords[-1].start_beat else None


def _group_lh_attacks(
    notes: list[NoteEvent],
    tolerance_sec: float = 0.075,
) -> list[list[NoteEvent]]:
    seq = sorted(
        [n for n in notes if n.hand == "left"],
        key=lambda n: (n.start_sec, n.midi_pitch),
    )
    groups: list[list[NoteEvent]] = []
    for n in seq:
        if not groups or n.start_sec - groups[-1][0].start_sec > tolerance_sec:
            groups.append([n])
        else:
            groups[-1].append(n)
    return groups


def repair_left_hand_chords(
    y: np.ndarray,
    sr: int,
    notes: list[NoteEvent],
    beat_times: list[float],
    key: KeyEstimate | None,
    chords: list[ChordEvent],
    *,
    hop_length: int = 512,
) -> tuple[list[NoteEvent], list[dict]]:
    """Conservatively repair dense left-hand chord attacks.

    The goal is precision, not a more elaborate arrangement. We only alter a
    chord when local audio evidence and the refined harmony agree strongly.
    Non-chord notes are never removed merely because the chord label disagrees.
    """
    out = deepcopy(notes)
    pitch_db, frame_times = _pitch_matrix(y, sr, hop_length=hop_length)
    onset_times = _onset_times(y, sr, hop_length)
    repairs: list[dict] = []

    # map copied notes by approximate identity so groups operate on `out`
    groups = _group_lh_attacks(out)

    for group in groups:
        if len(group) < 2:
            continue
        t = float(np.median([n.start_sec for n in group]))
        beat = time_to_beat(t, beat_times)
        chord = _chord_at(chords, beat)
        if chord is None:
            continue
        tones = {
            (chord.root_pc + iv) % 12
            for iv in QUALITY_INTERVALS.get(chord.quality, [0, 4, 7])
        }

        evidence: dict[int, dict] = {}
        for n in group:
            evidence[id(n)] = _pitch_evidence(
                pitch_db, frame_times, onset_times, n.midi_pitch, n.start_sec
            )
            if n.audio_support is None:
                n.audio_support = float(evidence[id(n)]["support"])
            if n.onset_support is None:
                n.onset_support = float(evidence[id(n)]["onset_support"])
            if n.pitch_margin is None:
                n.pitch_margin = float(evidence[id(n)]["pitch_margin_db"])

        # 1. Repair a weak non-chord hypothesis to a nearby missing chord tone
        # only when the replacement has overwhelming local spectral support.
        occupied = {n.midi_pitch for n in group}
        for n in list(group):
            cur_support = float(
                n.audio_support
                if n.audio_support is not None
                else evidence[id(n)]["support"]
            )
            if n.midi_pitch % 12 in tones or cur_support >= 0.72:
                continue

            options = []
            for p in range(max(28, n.midi_pitch - 7), min(64, n.midi_pitch + 7) + 1):
                if p in occupied or p % 12 not in tones:
                    continue
                e = _pitch_evidence(
                    pitch_db, frame_times, onset_times, p, n.start_sec
                )
                score = float(e["support"])
                if e["gain_db"] >= 2.5:
                    score += 0.04
                if e["pitch_margin_db"] >= 0.0:
                    score += 0.03
                options.append((score, p, e))
            if not options:
                continue
            score, p, e = max(options, key=lambda x: x[0])
            if (
                e["support"] >= 0.90
                and score >= cur_support + 0.28
                and e["onset_support"] >= 0.72
            ):
                old = n.midi_pitch
                occupied.discard(old)
                n.original_pitch = old if n.original_pitch is None else n.original_pitch
                n.midi_pitch = int(p)
                n.audio_support = float(e["support"])
                n.onset_support = float(e["onset_support"])
                n.pitch_margin = float(e["pitch_margin_db"])
                n.validation_status = "lh-chord-corrected"
                n.validation_reason = f"refined_harmony_{old}_to_{p}"
                occupied.add(p)
                repairs.append({
                    "time_sec": n.start_sec,
                    "action": "pitch_substitution",
                    "from_midi": old,
                    "to_midi": int(p),
                    "support": float(e["support"]),
                })

        # 2. Remove octave-duplicate ghosts before general chord pruning.
        # Transkun occasionally outputs a weak sub-octave or upper-octave copy
        # of the same struck bass key. Preserve real octaves when both notes are
        # strongly supported, but discard a clearly asymmetric duplicate.
        group.sort(key=lambda n: n.midi_pitch)
        remove_ids: set[int] = set()
        by_pc: dict[int, list[NoteEvent]] = defaultdict(list)
        for n in group:
            by_pc[n.midi_pitch % 12].append(n)
        for pc_group in by_pc.values():
            pc_group.sort(key=lambda n: n.midi_pitch)
            for low, high in zip(pc_group, pc_group[1:]):
                if high.midi_pitch - low.midi_pitch != 12:
                    continue
                ls = float(low.audio_support if low.audio_support is not None else 0.5)
                hs = float(high.audio_support if high.audio_support is not None else 0.5)
                lm = float(low.pitch_margin if low.pitch_margin is not None else 0.0)
                hm = float(high.pitch_margin if high.pitch_margin is not None else 0.0)
                if low.midi_pitch < 33 and ls < 0.30 and hs >= ls + 0.12:
                    remove_ids.add(id(low))
                elif ls < 0.55 and hs >= 0.88 and lm < 0.5:
                    remove_ids.add(id(low))
                elif hs < 0.60 and ls >= 0.88 and hm < 0.0:
                    remove_ids.add(id(high))

        # 3. Remove only clearly extraneous upper chord hypotheses. The lowest
        # credible surviving note receives strong protection because low
        # fundamentals are exactly where piano transcription is weakest.
        surviving_for_floor = [n for n in group if id(n) not in remove_ids]
        lowest = min(surviving_for_floor, key=lambda n: n.midi_pitch) if surviving_for_floor else group[0]
        chord_tone_count = sum(
            n.midi_pitch % 12 in tones
            for n in group
            if id(n) not in remove_ids
        )

        for n in group[1:]:
            if id(n) in remove_ids:
                continue
            support = float(n.audio_support if n.audio_support is not None else 0.5)
            margin = float(n.pitch_margin if n.pitch_margin is not None else 0.0)
            harmonic = float(
                n.harmonic_probability
                if n.harmonic_probability is not None
                else 0.0
            )
            is_tone = n.midi_pitch % 12 in tones
            interval = n.midi_pitch - lowest.midi_pitch

            alias_like = interval in {12, 19, 24, 28, 31, 36}
            if (
                not is_tone
                and len(group) >= 3
                and chord_tone_count >= 2
                and support < 0.68
            ):
                remove_ids.add(id(n))
            elif (
                not is_tone
                and alias_like
                and support < 0.80
                and (margin < 0.0 or harmonic >= 0.70)
            ):
                remove_ids.add(id(n))

        # Do not force ordinary left-hand chords down to three notes. Real
        # piano voicings frequently contain four notes (including octaves).
        # Only trim clusters larger than four, and keep one representative of
        # each detected chord pitch class before ranking any remaining colors.
        survivors = [n for n in group if id(n) not in remove_ids]
        if len(survivors) > 4:
            must_keep: list[NoteEvent] = []
            for pc in tones:
                pc_notes = [n for n in survivors if n.midi_pitch % 12 == pc]
                if pc_notes:
                    must_keep.append(max(
                        pc_notes,
                        key=lambda n: (
                            n.audio_support if n.audio_support is not None else n.confidence,
                            n.confidence,
                        ),
                    ))
            lowest_survivor = min(survivors, key=lambda n: n.midi_pitch)
            if lowest_survivor not in must_keep:
                must_keep.append(lowest_survivor)

            def rank(n: NoteEvent) -> float:
                support = float(
                    n.audio_support if n.audio_support is not None else n.confidence
                )
                tone = 0.28 if n.midi_pitch % 12 in tones else -0.08
                bass = 0.18 if n is lowest_survivor else 0.0
                margin = 0.025 * max(
                    -2.0, min(4.0, float(n.pitch_margin or 0.0))
                )
                return support + tone + bass + margin

            keep = {id(n) for n in must_keep}
            for n in sorted(survivors, key=rank, reverse=True):
                if len(keep) >= 4:
                    break
                keep.add(id(n))
            for n in survivors:
                if id(n) not in keep:
                    remove_ids.add(id(n))

        if remove_ids:
            for n in group:
                if id(n) in remove_ids:
                    repairs.append({
                        "time_sec": n.start_sec,
                        "action": "remove_extraneous_chord_note",
                        "midi_pitch": n.midi_pitch,
                        "audio_support": n.audio_support,
                    })
            out = [n for n in out if id(n) not in remove_ids]

    out.sort(key=lambda n: (n.start_sec, n.midi_pitch))
    return out, repairs
