from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from audio2score.models import (
    NoteEvent, ChordEvent, KeyEstimate, EngravingPlan,
    DynamicEvent, WedgeEvent, PedalEvent, PhraseSpan, SectionMarker,
)
from audio2score.export import write_piano_musicxml, write_piano_midi
from audio2score.engraving import detect_phrases


def test_detect_phrases_from_melodic_gaps():
    notes = [
        NoteEvent(0,0,72,start_beat=0,end_beat=1,hand='right',role='melody'),
        NoteEvent(0,0,74,start_beat=1,end_beat=2,hand='right',role='melody'),
        NoteEvent(0,0,76,start_beat=3,end_beat=4,hand='right',role='melody'),
        NoteEvent(0,0,77,start_beat=4,end_beat=5,hand='right',role='melody'),
    ]
    phrases=detect_phrases(notes,'4/4')
    assert len(phrases) == 2


def test_engraved_musicxml_contains_performance_marks(tmp_path: Path):
    notes = [
        NoteEvent(0,0,48,start_beat=0,end_beat=2,hand='left',role='bass'),
        NoteEvent(0,0,67,start_beat=0,end_beat=1,hand='right',role='melody'),
        NoteEvent(0,0,69,start_beat=1,end_beat=2,hand='right',role='melody'),
    ]
    plan=EngravingPlan(
        dynamics=[DynamicEvent(0,'mp')],
        wedges=[WedgeEvent(0.5,1.5,'crescendo')],
        pedals=[PedalEvent(0,2)],
        phrases=[PhraseSpan(0,2,67,69)],
        sections=[SectionMarker(0,'A','A',0)],
    )
    p=write_piano_musicxml(
        tmp_path/'x.musicxml', notes, [ChordEvent(0,4,0,'maj')],
        KeyEstimate(0,'major',1), 90, engraving_plan=plan,
    )
    root=ET.parse(p).getroot()
    assert root.find('.//dynamics/mp') is not None
    assert root.find('.//pedal') is not None
    assert root.find('.//rehearsal').text == 'A'
    assert root.find('.//slur[@type="start"]') is not None
    assert root.find('.//slur[@type="stop"]') is not None
    assert root.find('.//wedge[@type="crescendo"]') is not None


def test_piano_midi_contains_sustain_cc(tmp_path: Path):
    import pretty_midi
    notes=[
        NoteEvent(0,0,60,start_beat=0,end_beat=1,hand='left'),
        NoteEvent(0,0,72,start_beat=0,end_beat=1,hand='right'),
    ]
    plan=EngravingPlan(pedals=[PedalEvent(0,1)])
    p=write_piano_midi(tmp_path/'x.mid', notes, 120, plan)
    pm=pretty_midi.PrettyMIDI(str(p))
    assert all(any(cc.number == 64 for cc in inst.control_changes) for inst in pm.instruments)
