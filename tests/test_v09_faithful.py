from copy import deepcopy

from audio2score.arrange import make_faithful
from audio2score.models import NoteEvent


def test_faithful_preserves_validated_pitch_and_hand():
    notes = [
        NoteEvent(0.1, 1.0, 58, hand="left", confidence=0.9, start_beat=0.0, end_beat=1.0),
        NoteEvent(0.1, 1.0, 63, hand="right", confidence=0.9, start_beat=0.0, end_beat=1.0),
        NoteEvent(1.1, 1.5, 61, hand="left", confidence=0.9, start_beat=1.0, end_beat=1.5),
    ]
    out = make_faithful(deepcopy(notes), [])
    assert [(n.midi_pitch, n.hand) for n in out] == [
        (58, "left"), (63, "right"), (61, "left")
    ]


def test_faithful_does_not_drop_validated_nonchord_note():
    note = NoteEvent(
        0.2, 0.8, 65, hand="left", confidence=0.34,
        start_beat=0.0, end_beat=1.0
    )
    out = make_faithful([note], [])
    assert len(out) == 1
    assert out[0].midi_pitch == 65
