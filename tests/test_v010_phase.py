import numpy as np

from audio2score.models import NoteEvent
from audio2score.quantize import calibrate_notation_beat_times
from audio2score.rhythm import time_to_beat


def test_phase_calibration_reduces_notation_residual():
    beat_times = [i * 0.65 for i in range(40)]
    phase = -0.12
    # Attacks lie on a sixteenth lattice with a consistent beat-map phase error.
    notes = []
    for i in range(24):
        beat = i * 0.5 + phase
        t = beat * 0.65
        notes.append(NoteEvent(
            start_sec=t, end_sec=t + 0.25, midi_pitch=60 + (i % 5),
            confidence=0.95, audio_support=0.95, hand="right",
        ))
    corrected, applied_phase, info = calibrate_notation_beat_times(
        notes, beat_times, subdivisions=4
    )
    assert info["applied"]
    assert abs(applied_phase - phase) < 0.02

    before = []
    after = []
    for n in notes:
        b0 = time_to_beat(n.start_sec, beat_times)
        b1 = time_to_beat(n.start_sec, corrected)
        before.append(abs(b0 - round(b0 * 4) / 4))
        after.append(abs(b1 - round(b1 * 4) / 4))
    assert np.median(after) < np.median(before) * 0.35


def test_phase_calibration_leaves_aligned_grid_alone():
    beat_times = [i * 0.65 for i in range(30)]
    notes = [
        NoteEvent(
            start_sec=(i * 0.5) * 0.65,
            end_sec=(i * 0.5) * 0.65 + 0.2,
            midi_pitch=60,
            confidence=0.95,
            audio_support=0.95,
            hand="right",
        )
        for i in range(20)
    ]
    corrected, phase, info = calibrate_notation_beat_times(
        notes, beat_times, subdivisions=4
    )
    assert not info["applied"]
    assert phase == 0.0
    assert corrected == beat_times
