import numpy as np

from audio2score.models import NoteEvent, KeyEstimate
from audio2score.validation import validate_piano_transcription


def _tone(sr, seconds, freq, start, end, amp=0.7):
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    a, b = int(start * sr), int(end * sr)
    t = np.arange(max(0, b-a), dtype=np.float32) / sr
    sig = amp * np.sin(2*np.pi*freq*t)
    # short attack/release window
    n = len(sig)
    ramp = min(int(0.02*sr), max(1, n//4))
    if ramp > 1:
        sig[:ramp] *= np.linspace(0,1,ramp)
        sig[-ramp:] *= np.linspace(1,0,ramp)
    y[a:b] += sig[:b-a]
    return y


def test_audio_validation_prefers_present_pitch_over_false_neighbor():
    sr = 22050
    y = _tone(sr, 1.5, 440.0, 0.20, 0.85)
    beat_times = [0.0, 0.5, 1.0, 1.5]
    notes = [
        NoteEvent(0.20, 0.80, 69, confidence=0.95, hand="right"),  # A4, real
        NoteEvent(0.20, 0.80, 75, confidence=0.95, hand="right"),  # D#5, false
    ]
    result = validate_piano_transcription(
        y, sr, notes, beat_times,
        key=KeyEstimate(9, "minor", 1.0),
        chords=[], strength="balanced", add_missing=False,
        substitute_pitches=False, align=False,
    )
    real = next(d for d in result.diagnostics if d["midi_pitch"] == 69)
    false = next(d for d in result.diagnostics if d["midi_pitch"] == 75)
    assert real["audio_support"] > false["audio_support"]
    assert real["validation_status"] != "rejected"
