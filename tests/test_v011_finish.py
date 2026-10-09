import numpy as np

from audio2score.accuracy import _prefer_strong_raw_over_added_semitone
from audio2score.models import NoteEvent
from audio2score.quantize import refine_score_playback_beat_times


def test_raw_model_note_beats_added_semitone_when_both_are_strong():
    raw = NoteEvent(1.0, 2.0, 47, confidence=0.95, source="transkun", hand="left")
    raw.audio_support = 0.984
    added = NoteEvent(0.97, 1.7, 46, confidence=1.0, source="audio-validation-missing", hand="left")
    added.audio_support = 0.948
    choice = _prefer_strong_raw_over_added_semitone(raw, added)
    assert choice is not None
    kept, removed = choice
    assert kept.midi_pitch == 47
    assert removed.midi_pitch == 46


def test_weak_raw_note_can_still_yield_to_added_neighbor():
    raw = NoteEvent(1.0, 2.0, 47, confidence=0.4, source="transkun", hand="left")
    raw.audio_support = 0.31
    added = NoteEvent(1.0, 1.7, 46, confidence=0.98, source="audio-validation-missing", hand="left")
    added.audio_support = 0.98
    assert _prefer_strong_raw_over_added_semitone(raw, added) is None


def test_local_playback_refinement_reduces_attack_error():
    beat_times = [i * 0.65 for i in range(48)]
    notes = []
    # Most score attacks are close to grid; around beat 18 add a local 50 ms
    # performance anticipation similar to the remaining v0.10 benchmark case.
    for i in range(32):
        beat = i * 0.5
        start = beat * 0.65
        if beat >= 8.0:
            start -= 0.05
        n = NoteEvent(start, start + 0.2, 60 + (i % 4), confidence=0.95, hand="right")
        n.audio_support = 0.95
        n.start_beat = beat
        n.end_beat = beat + 0.25
        notes.append(n)
    refined, info = refine_score_playback_beat_times(notes, beat_times)
    assert info["applied"]
    assert info["median_abs_ms_after"] < info["median_abs_ms_before"]
