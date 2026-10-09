from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pretty_midi

from .models import NoteEvent, ChordEvent, KeyEstimate, EngravingPlan
from .music import (
    QUALITY_INTERVALS, key_fifths, midi_to_pitch_components,
    midi_to_pitch_components_for_key, pc_name, pc_name_for_key,
)

DIVISIONS = 4  # sixteenth-note resolution


def _duration_pieces(ticks: int) -> list[int]:
    allowed = [16, 12, 8, 6, 4, 3, 2, 1]
    pieces: list[int] = []
    left = int(ticks)
    while left > 0:
        d = next((x for x in allowed if x <= left), 1)
        pieces.append(d)
        left -= d
    return pieces


def _duration_pieces_at(start_tick: int, ticks: int, measure_start: int) -> list[int]:
    """Spell durations without hiding syncopation across a beat boundary."""
    allowed = [16, 12, 8, 6, 4, 3, 2, 1]
    pieces = []
    cur = int(start_tick)
    left = int(ticks)
    while left > 0:
        pos = (cur - measure_start) % DIVISIONS
        cap = left
        if pos != 0:
            cap = min(cap, DIVISIONS - pos)
        d = next((x for x in allowed if x <= cap), 1)
        pieces.append(d)
        cur += d
        left -= d
    return pieces


def _type_and_dots(ticks: int) -> tuple[str, int]:
    mapping = {
        16: ("whole", 0), 12: ("half", 1), 8: ("half", 0),
        6: ("quarter", 1), 4: ("quarter", 0), 3: ("eighth", 1),
        2: ("eighth", 0), 1: ("16th", 0),
    }
    return mapping[ticks]


def _add_note_xml(
    parent: ET.Element,
    pitch: int | None,
    ticks: int,
    tie_start: bool = False,
    tie_stop: bool = False,
    prefer_flats: bool = False,
    *,
    chord: bool = False,
    staff: int | None = None,
    voice: int = 1,
    fifths: int | None = None,
    slur_start: bool = False,
    slur_stop: bool = False,
) -> None:
    n = ET.SubElement(parent, "note")
    if chord:
        ET.SubElement(n, "chord")
    if pitch is None:
        ET.SubElement(n, "rest")
    else:
        step, alter, octave = (
            midi_to_pitch_components_for_key(pitch, fifths)
            if fifths is not None else midi_to_pitch_components(pitch, prefer_flats)
        )
        p = ET.SubElement(n, "pitch")
        ET.SubElement(p, "step").text = step
        if alter:
            ET.SubElement(p, "alter").text = str(alter)
        ET.SubElement(p, "octave").text = str(octave)
        if tie_stop:
            ET.SubElement(n, "tie", type="stop")
        if tie_start:
            ET.SubElement(n, "tie", type="start")
    ET.SubElement(n, "duration").text = str(ticks)
    ET.SubElement(n, "voice").text = str(voice)
    note_type, dots = _type_and_dots(ticks)
    ET.SubElement(n, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(n, "dot")
    if staff is not None:
        ET.SubElement(n, "staff").text = str(staff)

    if pitch is not None and (tie_start or tie_stop or slur_start or slur_stop):
        notations = ET.SubElement(n, "notations")
        if tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if tie_start:
            ET.SubElement(notations, "tied", type="start")
        if slur_stop:
            ET.SubElement(notations, "slur", type="stop", number="1")
        if slur_start:
            ET.SubElement(notations, "slur", type="start", number="1")


def _add_harmony_xml(
    parent: ET.Element,
    chord: ChordEvent,
    offset_ticks: int,
    prefer_flats: bool,
    fifths: int | None = None,
) -> None:
    h = ET.SubElement(parent, "harmony")
    root = ET.SubElement(h, "root")
    root_name = pc_name_for_key(chord.root_pc, fifths) if fifths is not None else pc_name(chord.root_pc, prefer_flats)
    ET.SubElement(root, "root-step").text = root_name[0]
    if len(root_name) > 1:
        ET.SubElement(root, "root-alter").text = "1" if root_name[1] == "#" else "-1"
    kind_map = {
        "maj": ("major", ""), "min": ("minor", "m"), "dim": ("diminished", "dim"),
        "aug": ("augmented", "+"), "sus2": ("suspended-second", "sus2"),
        "sus4": ("suspended-fourth", "sus4"), "7": ("dominant", "7"),
        "maj7": ("major-seventh", "maj7"), "min7": ("minor-seventh", "m7"),
    }
    kind_value, kind_text = kind_map.get(chord.quality, ("other", chord.quality))
    kind = ET.SubElement(h, "kind")
    kind.text = kind_value
    if kind_text:
        kind.set("text", kind_text)
    if chord.bass_pc is not None and chord.bass_pc != chord.root_pc:
        bass = ET.SubElement(h, "bass")
        bass_name = pc_name_for_key(chord.bass_pc, fifths) if fifths is not None else pc_name(chord.bass_pc, prefer_flats)
        ET.SubElement(bass, "bass-step").text = bass_name[0]
        if len(bass_name) > 1:
            ET.SubElement(bass, "bass-alter").text = "1" if bass_name[1] == "#" else "-1"
    if offset_ticks:
        ET.SubElement(h, "offset").text = str(offset_ticks)


def _parse_meter(meter: str) -> tuple[int, int]:
    try:
        beats, beat_type = meter.split("/", 1)
        beats_i, beat_type_i = int(beats), int(beat_type)
    except Exception as exc:
        raise ValueError(f"Invalid meter {meter!r}; expected e.g. 4/4 or 3/4") from exc
    if beat_type_i not in {2, 4, 8, 16} or beats_i <= 0:
        raise ValueError(f"Unsupported meter: {meter}")
    return beats_i, beat_type_i


def _add_offset(direction: ET.Element, offset_ticks: int) -> None:
    if offset_ticks:
        ET.SubElement(direction, "offset").text = str(max(0, int(offset_ticks)))


def _add_dynamic_direction(measure: ET.Element, mark: str, offset_ticks: int) -> None:
    direction = ET.SubElement(measure, "direction", placement="below")
    dt = ET.SubElement(direction, "direction-type")
    dynamics = ET.SubElement(dt, "dynamics")
    ET.SubElement(dynamics, mark)
    _add_offset(direction, offset_ticks)
    ET.SubElement(direction, "staff").text = "1"
    velocity = {"p": "48", "mp": "64", "mf": "80", "f": "96"}.get(mark, "80")
    ET.SubElement(direction, "sound", dynamics=velocity)


def _add_wedge_direction(measure: ET.Element, kind: str, start: bool, offset_ticks: int) -> None:
    direction = ET.SubElement(measure, "direction", placement="below")
    dt = ET.SubElement(direction, "direction-type")
    if start:
        ET.SubElement(dt, "wedge", type=kind)
    else:
        ET.SubElement(dt, "wedge", type="stop")
    _add_offset(direction, offset_ticks)
    ET.SubElement(direction, "staff").text = "1"


def _add_pedal_direction(measure: ET.Element, kind: str, offset_ticks: int) -> None:
    direction = ET.SubElement(measure, "direction", placement="below")
    dt = ET.SubElement(direction, "direction-type")
    ET.SubElement(dt, "pedal", type=kind, line="yes")
    _add_offset(direction, offset_ticks)
    ET.SubElement(direction, "staff").text = "2"


def _add_section_direction(measure: ET.Element, label: str, offset_ticks: int) -> None:
    direction = ET.SubElement(measure, "direction", placement="above")
    dt = ET.SubElement(direction, "direction-type")
    rehearsal = ET.SubElement(dt, "rehearsal", enclosure="square")
    rehearsal.text = label
    _add_offset(direction, offset_ticks)
    ET.SubElement(direction, "staff").text = "1"


def _measure_ranges(total_ticks: int, measure_ticks: int, pickup_ticks: int = 0):
    ranges = []
    if 0 < pickup_ticks < measure_ticks:
        ranges.append((0, pickup_ticks, "0", True))
        start = pickup_ticks
        number = 1
    else:
        start = 0
        number = 1
    while start < total_ticks:
        end = min(total_ticks, start + measure_ticks)
        if end <= start:
            break
        ranges.append((start, end, str(number), False))
        start = end
        number += 1
    if not ranges:
        ranges = [(0, measure_ticks, "1", False)]
    return ranges


def write_musicxml(
    output_path: str | Path,
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    key: KeyEstimate,
    tempo_bpm: float,
    meter: str = "4/4",
    title: str = "Transcription",
    composer: str = "",
) -> Path:
    """Legacy one-staff lead-sheet writer."""
    output_path = Path(output_path)
    beats_per_measure, beat_type = _parse_meter(meter)
    quarter_beats_per_measure = beats_per_measure * (4.0 / beat_type)
    measure_ticks = int(round(quarter_beats_per_measure * DIVISIONS))
    prefer_flats = key_fifths(key.tonic_pc, key.mode) < 0
    max_beat = max([0.0] + [n.end_beat or 0.0 for n in notes] + [c.end_beat for c in chords])
    total_ticks = max(measure_ticks, int(round(max_beat * DIVISIONS)))
    ranges = _measure_ranges(total_ticks, measure_ticks, 0)

    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    identification = ET.SubElement(score, "identification")
    if composer:
        ET.SubElement(identification, "creator", type="composer").text = composer
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "Audio2Score 0.9.0"
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Melody"
    part = ET.SubElement(score, "part", id="P1")

    note_ticks = []
    for n in notes:
        s = int(round((n.start_beat or 0.0) * DIVISIONS))
        e = int(round((n.end_beat or 0.0) * DIVISIONS))
        if e > s:
            note_ticks.append((s, e, n.midi_pitch))
    note_ticks.sort()

    for mi, (ms, me, number, implicit) in enumerate(ranges):
        attrs = {"number": number}
        if implicit:
            attrs["implicit"] = "yes"
        measure = ET.SubElement(part, "measure", **attrs)
        if mi == 0:
            a = ET.SubElement(measure, "attributes")
            ET.SubElement(a, "divisions").text = str(DIVISIONS)
            key_el = ET.SubElement(a, "key")
            ET.SubElement(key_el, "fifths").text = str(key_fifths(key.tonic_pc, key.mode))
            time_el = ET.SubElement(a, "time")
            ET.SubElement(time_el, "beats").text = str(beats_per_measure)
            ET.SubElement(time_el, "beat-type").text = str(beat_type)
            clef = ET.SubElement(a, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            direction = ET.SubElement(measure, "direction", placement="above")
            dt = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = f"{tempo_bpm:.2f}"
            ET.SubElement(direction, "sound", tempo=f"{tempo_bpm:.4f}")
        for c in chords:
            ct = int(round(c.start_beat * DIVISIONS))
            if ms <= ct < me:
                _add_harmony_xml(measure, c, ct-ms, prefer_flats)
        cursor = ms
        for s, e, p in [(s,e,p) for s,e,p in note_ticks if e > ms and s < me]:
            ss, ee = max(s, ms), min(e, me)
            if ss > cursor:
                cur = cursor
                for d in _duration_pieces_at(cur, ss-cur, ms):
                    _add_note_xml(measure, None, d, prefer_flats=prefer_flats)
                    cur += d
                cursor = ss
            if ss < cursor:
                ss = cursor
            if ee <= ss:
                continue
            cur = ss
            pieces = _duration_pieces_at(ss, ee-ss, ms)
            for i, d in enumerate(pieces):
                _add_note_xml(
                    measure, p, d,
                    tie_start=(e > ee or i < len(pieces)-1),
                    tie_stop=(s < ss or i > 0),
                    prefer_flats=prefer_flats,
                )
                cur += d
            cursor = ee
        if cursor < me:
            cur = cursor
            for d in _duration_pieces_at(cur, me-cursor, ms):
                _add_note_xml(measure, None, d, prefer_flats=prefer_flats)
                cur += d

    ET.indent(score, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(score).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _staff_segments(note_ticks: list[tuple[int,int,int]], m_start: int, m_end: int):
    relevant = [(max(s,m_start), min(e,m_end), p, s, e) for s,e,p in note_ticks if e > m_start and s < m_end]
    boundaries = {m_start, m_end}
    for ss, ee, _, _, _ in relevant:
        boundaries.add(ss)
        boundaries.add(ee)
    b = sorted(boundaries)
    segments = []
    for a, z in zip(b, b[1:]):
        if z <= a:
            continue
        active = sorted({p for ss,ee,p,_,_ in relevant if ss <= a and ee >= z})
        segments.append((a, z, active))
    return segments


def _write_staff_stream(
    measure: ET.Element,
    segments,
    staff: int,
    voice: int,
    prefer_flats: bool,
    fifths: int | None = None,
    *,
    measure_start: int,
    phrase_starts: set[tuple[int,int]] | None = None,
    phrase_ends: set[tuple[int,int]] | None = None,
):
    phrase_starts = phrase_starts or set()
    phrase_ends = phrase_ends or set()

    for i, (a, z, pitches) in enumerate(segments):
        ticks = z - a
        prev = set(segments[i-1][2]) if i > 0 else set()
        nxt = set(segments[i+1][2]) if i + 1 < len(segments) else set()
        if not pitches:
            cur = a
            for d in _duration_pieces_at(cur, ticks, measure_start):
                _add_note_xml(
                    measure, None, d, prefer_flats=prefer_flats,
                    staff=staff, voice=voice, fifths=fifths,
                )
                cur += d
            continue

        pieces = _duration_pieces_at(a, ticks, measure_start)
        cur = a
        for j, d in enumerate(pieces):
            piece_end = cur + d
            for k, p in enumerate(pitches):
                tie_stop = (j > 0) or (p in prev)
                tie_start = (j < len(pieces)-1) or (p in nxt)
                slur_start = (j == 0 and (a, p) in phrase_starts)
                slur_stop = (j == len(pieces)-1 and (z, p) in phrase_ends)
                _add_note_xml(
                    measure, p, d,
                    tie_start=tie_start,
                    tie_stop=tie_stop,
                    prefer_flats=prefer_flats,
                    chord=(k > 0),
                    staff=staff,
                    voice=voice,
                    fifths=fifths,
                    slur_start=slur_start,
                    slur_stop=slur_stop,
                )
            cur = piece_end


def _events_in_measure(plan: EngravingPlan, ms: int, me: int, is_last: bool = False):
    if not plan:
        return [], [], [], [], []
    dynamics = [(int(round(x.beat*DIVISIONS)), x) for x in plan.dynamics if ms <= int(round(x.beat*DIVISIONS)) < me]
    section = [(int(round(x.beat*DIVISIONS)), x) for x in plan.sections if ms <= int(round(x.beat*DIVISIONS)) < me]

    wedges = []
    for x in plan.wedges:
        s, e = int(round(x.start_beat*DIVISIONS)), int(round(x.end_beat*DIVISIONS))
        if ms <= s < me:
            wedges.append((s, x.kind, True))
        if ms <= e < me or (is_last and e == me):
            wedges.append((e, x.kind, False))

    pedals = []
    for x in plan.pedals:
        s, e = int(round(x.start_beat*DIVISIONS)), int(round(x.end_beat*DIVISIONS))
        if ms <= s < me:
            pedals.append((s, "start"))
        if ms <= e < me or (is_last and e == me):
            pedals.append((e, "stop"))
    return dynamics, section, wedges, pedals, []


def write_piano_musicxml(
    output_path: str | Path,
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    key: KeyEstimate,
    tempo_bpm: float,
    meter: str = "4/4",
    title: str = "Piano Transcription",
    composer: str = "",
    engraving_plan: EngravingPlan | None = None,
) -> Path:
    """Write a two-staff grand-staff MusicXML score with engraving hints."""
    output_path = Path(output_path)
    beats_per_measure, beat_type = _parse_meter(meter)
    quarter_beats_per_measure = beats_per_measure * (4.0 / beat_type)
    measure_ticks = int(round(quarter_beats_per_measure * DIVISIONS))
    fifths_value = key_fifths(key.tonic_pc, key.mode)
    prefer_flats = fifths_value < 0
    max_beat = max([0.0] + [n.end_beat or 0.0 for n in notes] + [c.end_beat for c in chords])
    total_ticks = max(measure_ticks, int(round(max_beat * DIVISIONS)))
    pickup_ticks = 0
    if engraving_plan and 0 < engraving_plan.pickup_beats < quarter_beats_per_measure:
        pickup_ticks = int(round(engraving_plan.pickup_beats * DIVISIONS))
    ranges = _measure_ranges(total_ticks, measure_ticks, pickup_ticks)

    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    identification = ET.SubElement(score, "identification")
    if composer:
        ET.SubElement(identification, "creator", type="composer").text = composer
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "Audio2Score 0.9.0"
    part_list = ET.SubElement(score, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Piano"
    si = ET.SubElement(sp, "score-instrument", id="P1-I1")
    ET.SubElement(si, "instrument-name").text = "Piano"
    mi = ET.SubElement(sp, "midi-instrument", id="P1-I1")
    ET.SubElement(mi, "midi-channel").text = "1"
    ET.SubElement(mi, "midi-program").text = "1"
    part = ET.SubElement(score, "part", id="P1")

    right, left = [], []
    for n in notes:
        s = int(round((n.start_beat or 0.0) * DIVISIONS))
        e = int(round((n.end_beat or 0.0) * DIVISIONS))
        if e <= s:
            continue
        (left if n.hand == "left" else right).append((s, e, int(n.midi_pitch)))
    right.sort()
    left.sort()

    phrase_starts = set()
    phrase_ends = set()
    if engraving_plan:
        for p in engraving_plan.phrases:
            phrase_starts.add((int(round(p.start_beat * DIVISIONS)), int(p.start_pitch)))
            phrase_ends.add((int(round(p.end_beat * DIVISIONS)), int(p.end_pitch)))

    for m, (ms, me, number, implicit) in enumerate(ranges):
        attrs = {"number": number}
        if implicit:
            attrs["implicit"] = "yes"
        measure = ET.SubElement(part, "measure", **attrs)
        if m == 0:
            a = ET.SubElement(measure, "attributes")
            ET.SubElement(a, "divisions").text = str(DIVISIONS)
            key_el = ET.SubElement(a, "key")
            ET.SubElement(key_el, "fifths").text = str(fifths_value)
            time_el = ET.SubElement(a, "time")
            ET.SubElement(time_el, "beats").text = str(beats_per_measure)
            ET.SubElement(time_el, "beat-type").text = str(beat_type)
            ET.SubElement(a, "staves").text = "2"
            clef1 = ET.SubElement(a, "clef", number="1")
            ET.SubElement(clef1, "sign").text = "G"
            ET.SubElement(clef1, "line").text = "2"
            clef2 = ET.SubElement(a, "clef", number="2")
            ET.SubElement(clef2, "sign").text = "F"
            ET.SubElement(clef2, "line").text = "4"
            direction = ET.SubElement(measure, "direction", placement="above")
            dt = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = f"{tempo_bpm:.2f}"
            ET.SubElement(direction, "sound", tempo=f"{tempo_bpm:.4f}")

        for c in chords:
            ct = int(round(c.start_beat * DIVISIONS))
            if ms <= ct < me:
                _add_harmony_xml(measure, c, ct-ms, prefer_flats, fifths=fifths_value)

        if engraving_plan:
            dynamics, sections, wedges, pedals, _ = _events_in_measure(engraving_plan, ms, me, is_last=(m == len(ranges)-1))
            for tick, ev in sorted(sections, key=lambda x: x[0]):
                _add_section_direction(measure, ev.label, tick-ms)
            for tick, ev in sorted(dynamics, key=lambda x: x[0]):
                _add_dynamic_direction(measure, ev.mark, tick-ms)
            for tick, kind, is_start in sorted(wedges, key=lambda x: (x[0], not x[2])):
                _add_wedge_direction(measure, kind, is_start, tick-ms)
            for tick, kind in sorted(pedals, key=lambda x: x[0]):
                _add_pedal_direction(measure, kind, tick-ms)

        rseg = _staff_segments(right, ms, me)
        lseg = _staff_segments(left, ms, me)
        if not rseg:
            rseg = [(ms, me, [])]
        if not lseg:
            lseg = [(ms, me, [])]
        _write_staff_stream(
            measure, rseg, 1, 1, prefer_flats, fifths=fifths_value,
            measure_start=ms, phrase_starts=phrase_starts, phrase_ends=phrase_ends,
        )
        backup = ET.SubElement(measure, "backup")
        ET.SubElement(backup, "duration").text = str(me-ms)
        _write_staff_stream(
            measure, lseg, 2, 2, prefer_flats, fifths=fifths_value,
            measure_start=ms,
        )

    ET.indent(score, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(score).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def write_midi(
    output_path: str | Path,
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    tempo_bpm: float,
    include_chords: bool = True,
) -> Path:
    output_path = Path(output_path)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    spb = 60.0 / float(tempo_bpm)
    melody = pretty_midi.Instrument(program=0, name="Melody")
    for n in notes:
        if n.start_beat is None or n.end_beat is None:
            continue
        s, e = n.start_beat*spb, n.end_beat*spb
        if e > s:
            melody.notes.append(pretty_midi.Note(
                velocity=int(n.velocity), pitch=int(n.midi_pitch), start=s, end=e
            ))
    pm.instruments.append(melody)
    if include_chords:
        harmony = pretty_midi.Instrument(program=0, name="Chord guide")
        for c in chords:
            s, e = c.start_beat*spb, c.end_beat*spb
            intervals = QUALITY_INTERVALS.get(c.quality, [0,4,7])
            root = 48 + c.root_pc
            if root > 55:
                root -= 12
            for iv in intervals:
                harmony.notes.append(pretty_midi.Note(
                    velocity=42, pitch=root+iv, start=s, end=e
                ))
        pm.instruments.append(harmony)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    return output_path



def write_performance_midi(
    output_path: str | Path,
    notes: list[NoteEvent],
    tempo_bpm: float = 120.0,
    source_midi: str | Path | None = None,
    *,
    track_suffix: str = " - validated performance",
) -> Path:
    """Write unquantized validated piano performance MIDI in real seconds.

    This deliberately does *not* use start_beat/end_beat. Score quantization is
    a later representation. If a raw source MIDI is available, sustain CC64
    events are copied into both hand tracks so performance playback retains the
    backend's pedal information where possible.
    """
    output_path = Path(output_path)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    rh = pretty_midi.Instrument(program=0, name=f"Right hand{track_suffix}")
    lh = pretty_midi.Instrument(program=0, name=f"Left hand{track_suffix}")
    for n in notes:
        s = max(0.0, float(n.start_sec))
        e = max(s + 0.015, float(n.end_sec))
        target = lh if n.hand == "left" else rh
        target.notes.append(pretty_midi.Note(
            velocity=int(np_clip(n.velocity, 1, 127)),
            pitch=int(n.midi_pitch), start=s, end=e,
        ))

    if source_midi:
        try:
            src = pretty_midi.PrettyMIDI(str(source_midi))
            pedal = []
            for inst in src.instruments:
                pedal.extend(cc for cc in inst.control_changes if cc.number == 64)
            pedal.sort(key=lambda cc: cc.time)
            for cc in pedal:
                for inst in (rh, lh):
                    inst.control_changes.append(pretty_midi.ControlChange(64, cc.value, cc.time))
        except Exception:
            pass

    pm.instruments.extend([rh, lh])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    return output_path

def _beat_to_seconds(
    beat: float,
    beat_times: list[float] | None,
    tempo_bpm: float,
) -> float:
    """Map notated beat position back onto the performed timeline.

    v0.7 wrote score MIDI with one average tempo, which accumulated more than a
    second of drift in expressive piano recordings. v0.8 keeps quantized beat
    locations for notation but renders them through the locally varying source
    beat map for playback.
    """
    if beat_times and len(beat_times) >= 2:
        bt = [float(x) for x in beat_times]
        if beat <= 0:
            period = bt[1] - bt[0]
            return float(bt[0] + beat * period)
        last_index = len(bt) - 1
        if beat >= last_index:
            period = bt[-1] - bt[-2]
            return float(bt[-1] + (beat - last_index) * period)
        lo = int(math.floor(beat))
        frac = float(beat - lo)
        return float(bt[lo] + frac * (bt[lo + 1] - bt[lo]))
    return float(beat) * 60.0 / float(tempo_bpm)


def write_piano_midi(
    output_path: str | Path,
    notes: list[NoteEvent],
    tempo_bpm: float,
    engraving_plan: EngravingPlan | None = None,
    beat_times: list[float] | None = None,
    preserve_source_durations: bool = False,
) -> Path:
    output_path = Path(output_path)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    rh = pretty_midi.Instrument(program=0, name="Right hand")
    lh = pretty_midi.Instrument(program=0, name="Left hand")
    for n in notes:
        if n.start_beat is None or n.end_beat is None:
            continue
        s = _beat_to_seconds(float(n.start_beat), beat_times, tempo_bpm)
        if (
            preserve_source_durations
            and n.end_sec > n.start_sec
            and n.source
            and not n.source.startswith("arranged-")
        ):
            # Quantized attack, original validated key-release duration.
            raw_dur = max(0.02, float(n.end_sec - n.start_sec))
            e = s + raw_dur
        else:
            e = _beat_to_seconds(float(n.end_beat), beat_times, tempo_bpm)
        if e <= s:
            continue
        target = lh if n.hand == "left" else rh
        target.notes.append(pretty_midi.Note(
            velocity=int(np_clip(n.velocity,1,127)),
            pitch=int(n.midi_pitch), start=max(0.0, s), end=max(s + 0.015, e),
        ))

    if engraving_plan:
        for ped in engraving_plan.pedals:
            s = _beat_to_seconds(float(ped.start_beat), beat_times, tempo_bpm)
            e = _beat_to_seconds(float(ped.end_beat), beat_times, tempo_bpm)
            for inst in (rh, lh):
                inst.control_changes.append(
                    pretty_midi.ControlChange(64, 127, max(0.0, s))
                )
                inst.control_changes.append(
                    pretty_midi.ControlChange(64, 0, max(s + 0.02, e))
                )

    pm.instruments.extend([rh, lh])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    return output_path


def np_clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def find_musescore() -> str | None:
    for name in ("MuseScore4.exe", "MuseScore4", "mscore", "musescore"):
        found = shutil.which(name)
        if found:
            return found
    for p in [
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore Studio 4\bin\MuseScore4.exe"),
    ]:
        if p.exists():
            return str(p)
    return None


def export_pdf_with_musescore(
    musicxml_path: str | Path,
    pdf_path: str | Path,
) -> Path | None:
    exe = find_musescore()
    if not exe:
        return None
    pdf_path = Path(pdf_path)
    subprocess.run([exe, "-o", str(pdf_path), str(musicxml_path)], check=True)
    return pdf_path if pdf_path.exists() else None
