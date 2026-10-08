from pathlib import Path
import xml.etree.ElementTree as ET

from audio2score.export import write_piano_musicxml
from audio2score.models import NoteEvent, ChordEvent, KeyEstimate


def test_grand_staff_musicxml(tmp_path: Path):
    notes=[
        NoteEvent(0,1,48,start_beat=0,end_beat=2,hand="left"),
        NoteEvent(0,1,55,start_beat=0,end_beat=2,hand="left"),
        NoteEvent(0,0.5,67,start_beat=0,end_beat=1,hand="right"),
        NoteEvent(0,0.5,70,start_beat=0,end_beat=1,hand="right"),
    ]
    path=tmp_path/"piano.musicxml"
    write_piano_musicxml(path,notes,[ChordEvent(0,4,0,"maj")],KeyEstimate(0,"major",1),90)
    tree=ET.parse(path)
    root=tree.getroot()
    assert root.find('.//staves').text == '2'
    assert {x.text for x in root.findall('.//staff')} == {'1','2'}
