from audio2score.models import NoteEvent
from audio2score.quantize import adaptive_quantize_polyphonic_notes


def test_adaptive_quantizer_maps_human_quarters_without_changing_seconds():
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    notes = [
        NoteEvent(0.012, 0.46, 60, hand="right"),
        NoteEvent(0.512, 0.96, 62, hand="right"),
        NoteEvent(1.018, 1.46, 64, hand="right"),
    ]
    original = [(n.start_sec, n.end_sec) for n in notes]
    q = adaptive_quantize_polyphonic_notes(notes, beat_times, max_subdivisions=4)
    assert [round(n.start_beat, 5) for n in q] == [0.0, 1.0, 2.0]
    assert [(n.start_sec, n.end_sec) for n in notes] == original
