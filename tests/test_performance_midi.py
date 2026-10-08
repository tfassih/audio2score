import pretty_midi

from audio2score.models import NoteEvent
from audio2score.export import write_performance_midi


def test_performance_midi_preserves_unquantized_seconds(tmp_path):
    p = tmp_path / "performance.mid"
    notes = [
        NoteEvent(0.137, 0.693, 60, velocity=73, hand="right"),
        NoteEvent(0.229, 1.111, 43, velocity=65, hand="left"),
    ]
    write_performance_midi(p, notes, tempo_bpm=92.3)
    pm = pretty_midi.PrettyMIDI(str(p))
    starts = sorted(n.start for inst in pm.instruments for n in inst.notes)
    assert abs(starts[0] - 0.137) < 0.002
    assert abs(starts[1] - 0.229) < 0.002
