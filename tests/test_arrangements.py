from audio2score.arrange import build_arrangements
from audio2score.models import NoteEvent, ChordEvent


def test_arrangements_clean_and_infer_bass():
    notes = [
        NoteEvent(0,1,39,0.95,start_beat=0,end_beat=4,hand='left'),
        NoteEvent(0,1,46,0.90,start_beat=0,end_beat=4,hand='left'),
        NoteEvent(0,1,63,0.95,start_beat=0,end_beat=4,hand='right'),
        NoteEvent(0,1,66,0.90,start_beat=0,end_beat=4,hand='right'),
        NoteEvent(0,1,70,0.95,start_beat=0,end_beat=1,hand='right'),
        NoteEvent(0,1,82,0.30,start_beat=0,end_beat=1,hand='right'),
    ]
    chords=[ChordEvent(0,4,3,'min')]
    a=build_arrangements(notes,chords)
    assert a.chords[0].bass_pc == 3
    assert len(a.faithful) == len(notes)
    assert any(n.role == 'melody' for n in a.faithful)
    assert any(n.role == 'bass' for n in a.faithful)
    assert all(n.hand in {'left','right'} for n in a.easy)
    # v0.5 Intermediate deliberately generates a regular broken-chord LH.
    lh = sorted([n for n in a.intermediate if n.hand == 'left'], key=lambda n:n.start_beat)
    assert [n.start_beat for n in lh[:4]] == [0.0, 1.0, 2.0, 3.0]


def test_easy_uses_half_note_bass_then_fifth():
    notes=[NoteEvent(0,1,72,0.9,start_beat=0,end_beat=1,hand='right')]
    chords=[ChordEvent(0,8,0,'maj')]
    a=build_arrangements(notes,chords)
    lh=sorted([n for n in a.easy if n.hand=='left'], key=lambda n:n.start_beat)
    assert [n.start_beat for n in lh] == [0,2,4,6]
    assert [n.role for n in lh] == ['bass','accompaniment','bass','accompaniment']


def test_global_melody_tracking_rejects_weak_octave_spike():
    notes = [
        NoteEvent(0,1,72,0.95,start_beat=0,end_beat=0.5,hand='right'),
        NoteEvent(0,1,74,0.92,start_beat=1,end_beat=1.5,hand='right'),
        NoteEvent(0,1,86,0.35,start_beat=1,end_beat=1.5,hand='right'),
        NoteEvent(0,1,76,0.93,start_beat=2,end_beat=2.5,hand='right'),
    ]
    chords=[ChordEvent(0,4,0,'maj')]
    a=build_arrangements(notes,chords)
    melody=[n for n in a.faithful if n.role=='melody']
    assert [n.midi_pitch for n in melody] == [72,74,76]
