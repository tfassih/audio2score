from pathlib import Path
import xml.etree.ElementTree as ET

from audio2score.models import NoteEvent, ChordEvent, KeyEstimate
from audio2score.export import write_musicxml


def test_musicxml_is_well_formed(tmp_path: Path):
    notes = [
        NoteEvent(0, 0, 60, start_beat=0.0, end_beat=1.0),
        NoteEvent(0, 0, 64, start_beat=1.0, end_beat=2.0),
        NoteEvent(0, 0, 67, start_beat=2.0, end_beat=4.0),
    ]
    chords = [ChordEvent(0, 2, 0, "maj"), ChordEvent(2, 4, 7, "7")]
    p = write_musicxml(tmp_path / "test.musicxml", notes, chords, KeyEstimate(0, "major", 1.0), 120.0)
    tree = ET.parse(p)
    assert tree.getroot().tag == "score-partwise"
    assert tree.find(".//harmony") is not None
