from pathlib import Path

import pretty_midi
import pytest

from audio2score.accuracy import refine_harmony_from_piano_notes
from audio2score.export import _beat_to_seconds, write_piano_midi
from audio2score.models import KeyEstimate, NoteEvent


def test_beat_to_seconds_uses_local_performance_map():
    beats = [0.0, 0.50, 1.10, 1.80]
    assert _beat_to_seconds(1.5, beats, 120.0) == pytest.approx(0.80, abs=1e-6)


def test_score_midi_preserves_variable_beat_timing(tmp_path: Path):
    beats = [0.0, 0.50, 1.10, 1.80]
    note = NoteEvent(
        0.0, 0.0, 60,
        start_beat=1.0, end_beat=2.0,
        hand="right", velocity=80,
    )
    path = tmp_path / "mapped.mid"
    write_piano_midi(path, [note], 120.0, beat_times=beats)
    pm = pretty_midi.PrettyMIDI(str(path))
    n = pm.instruments[0].notes[0]
    assert n.start == pytest.approx(0.50, abs=0.005)
    assert n.end == pytest.approx(1.10, abs=0.005)


def test_note_driven_harmony_separates_adjacent_bars():
    # 100 BPM-like synthetic beat map, 2 bars.
    beats = [0.6 * i for i in range(10)]
    notes = [
        # D# minor bar
        NoteEvent(0.05, 2.20, 39, hand="left", audio_support=0.95, velocity=70),
        NoteEvent(0.06, 2.20, 46, hand="left", audio_support=0.95, velocity=65),
        NoteEvent(0.07, 2.20, 54, hand="left", audio_support=0.90, velocity=62),
        # B major bar
        NoteEvent(2.45, 4.70, 47, hand="left", audio_support=0.96, velocity=72),
        NoteEvent(2.46, 4.70, 54, hand="left", audio_support=0.92, velocity=64),
        NoteEvent(2.47, 4.70, 51, hand="left", audio_support=0.90, velocity=60),
    ]
    chords = refine_harmony_from_piano_notes(
        notes, beats, KeyEstimate(3, "minor", 0.8), beats_per_bar=4
    )
    assert chords[0].root_pc == 3 and chords[0].quality == "min"
    second = next(c for c in chords if c.start_beat <= 4 < c.end_beat)
    assert second.root_pc == 11 and second.quality == "maj"
