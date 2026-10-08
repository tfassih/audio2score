from audio2score.models import NoteEvent
from audio2score.rhythm import analyze_rhythm
from audio2score.validation import _repair_performance_gaps


def test_gap_repair_removes_tiny_artificial_pause():
    notes = [
        NoteEvent(0.0, 0.45, 60, hand="right"),
        NoteEvent(0.52, 1.0, 64, hand="right"),
    ]
    out = _repair_performance_gaps(notes, onset_times=__import__("numpy").array([0.0, 0.52]))
    assert max(n.end_sec for n in out if n.start_sec == 0.0) > 0.45


def test_gap_repair_keeps_large_rest():
    notes = [
        NoteEvent(0.0, 0.35, 60, hand="right"),
        NoteEvent(0.70, 1.0, 64, hand="right"),
    ]
    out = _repair_performance_gaps(notes, onset_times=__import__("numpy").array([0.0, 0.70]))
    assert max(n.end_sec for n in out if n.start_sec == 0.0) == 0.35
