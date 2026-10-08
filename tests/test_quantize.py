from audio2score.models import NoteEvent
from audio2score.quantize import quantize_notes


def test_quantize_to_sixteenths():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    notes = [NoteEvent(0.13, 0.62, 60, 0.9)]
    out = quantize_notes(notes, beats, subdivisions_per_beat=4)
    assert len(out) == 1
    assert out[0].start_beat == 0.25
    assert out[0].end_beat == 1.25
