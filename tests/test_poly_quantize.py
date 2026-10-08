from audio2score.models import NoteEvent
from audio2score.quantize import quantize_polyphonic_notes


def test_polyphony_survives_quantization():
    beats=[0,1,2,3,4]
    notes=[
        NoteEvent(0.02,0.9,60,hand="right"),
        NoteEvent(0.03,0.9,64,hand="right"),
        NoteEvent(0.04,0.9,67,hand="right"),
    ]
    out=quantize_polyphonic_notes(notes,beats,4)
    assert len(out)==3
    assert {n.midi_pitch for n in out} == {60,64,67}
    assert len({n.start_beat for n in out}) == 1
