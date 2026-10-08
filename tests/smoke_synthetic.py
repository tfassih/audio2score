"""Generate a tiny synthetic melody/harmony WAV and run the whole baseline pipeline.
Not part of normal pytest because it is intentionally signal-processing heavy.
"""
from pathlib import Path
import numpy as np
import soundfile as sf

from audio2score.pipeline import transcribe_song

SR = 22050
BPM = 120
BEAT = 60 / BPM


def tone(freq, dur, amp=0.2):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (dur - t) / 0.04)
    env = np.clip(env, 0, 1)
    return amp * np.sin(2 * np.pi * freq * t) * env


def midi_hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def main():
    out = Path("synthetic_smoke")
    out.mkdir(exist_ok=True)
    # 8 beats: C major for four, G major for four; melody C D E G | G A G E
    melody = [60, 62, 64, 67, 67, 69, 67, 64]
    audio = []
    for i, m in enumerate(melody):
        chord = [48, 52, 55] if i < 4 else [43, 47, 50]
        seg = tone(midi_hz(m), BEAT, 0.38)
        for c in chord:
            seg += tone(midi_hz(c), BEAT, 0.06)
        audio.append(seg)
    y = np.concatenate(audio).astype(np.float32)
    wav = out / "synthetic.wav"
    sf.write(wav, y, SR)
    result = transcribe_song(wav, out / "result", meter="4/4", melody_backend="pyin", make_pdf=False)
    print(result["outputs"])


if __name__ == "__main__":
    main()
