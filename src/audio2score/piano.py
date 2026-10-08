from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import logging
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import pretty_midi
from scipy.signal import find_peaks

from .models import NoteEvent

log = logging.getLogger(__name__)


def _midi_velocity(peak_db: float, strongest_db: float) -> int:
    # Preserve some dynamics without letting spectral amplitude dominate.
    rel = float(np.clip((peak_db - (strongest_db - 28.0)) / 28.0, 0.0, 1.0))
    return int(round(46 + 60 * rel))


def _read_midi_notes(midi_path: str | Path, source: str) -> list[NoteEvent]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[NoteEvent] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            if n.end <= n.start:
                continue
            notes.append(
                NoteEvent(
                    start_sec=float(n.start),
                    end_sec=float(n.end),
                    midi_pitch=int(n.pitch),
                    confidence=0.95,
                    source=source,
                    velocity=int(n.velocity),
                )
            )
    return sorted(notes, key=lambda n: (n.start_sec, n.midi_pitch))


def transcribe_with_transkun(
    audio_path: str | Path,
    device: str = "cpu",
) -> list[NoteEvent]:
    """Use the optional Transkun CLI if it is installed.

    Transkun is intentionally optional: v0.4 must remain runnable without a
    heavyweight neural model. Install it separately, then select
    --piano-backend transkun (or auto).
    """
    exe = shutil.which("transkun")
    if not exe:
        raise RuntimeError("Transkun executable not found on PATH")
    with tempfile.TemporaryDirectory(prefix="audio2score-transkun-") as td:
        midi_path = Path(td) / "transkun.mid"
        cmd = [exe, str(audio_path), str(midi_path)]
        if device:
            cmd += ["--device", device]
        subprocess.run(cmd, check=True)
        if not midi_path.exists():
            raise RuntimeError("Transkun completed without producing MIDI")
        return _read_midi_notes(midi_path, "transkun")


def transcribe_piano_polyphonic_spectral(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    bins_per_semitone: int = 3,
    min_midi: int = 28,
    max_midi: int = 88,
    max_notes_per_attack: int = 7,
) -> list[NoteEvent]:
    """Deterministic polyphonic piano fallback.

    The detector uses a high-resolution CQT, global attack detection,
    semitone-local spectral peaks, harmonic suppression and per-pitch decay
    tracking. It is not a replacement for a modern neural piano model, but it
    produces useful two-hand scaffolding on clean solo-piano recordings and
    requires only the baseline Audio2Score dependencies.
    """
    if y.size == 0:
        return []

    bpo = 12 * bins_per_semitone
    piano_min = 21  # A0
    piano_max = 108 # C8
    n_semitones = piano_max - piano_min + 1
    n_bins = n_semitones * bins_per_semitone

    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=librosa.midi_to_hz(piano_min),
            n_bins=n_bins,
            bins_per_octave=bpo,
            filter_scale=1.0,
        )
    )
    db = librosa.amplitude_to_db(C, ref=np.max)

    # Collapse the sub-semitone CQT bins into one salience curve per key.
    pitch_db = np.full((n_semitones, db.shape[1]), -80.0, dtype=np.float32)
    for i in range(n_semitones):
        center = i * bins_per_semitone
        lo = max(0, center - 1)
        hi = min(db.shape[0], center + 2)
        pitch_db[i] = np.max(db[lo:hi], axis=0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
        delta=0.12,
        wait=2,
    ).astype(int)
    first_quarter_sec = max(1, int(round(0.25 * sr / hop_length)))
    if onset_frames.size == 0 or onset_frames[0] > first_quarter_sec:
        onset_frames = np.insert(onset_frames, 0, 0)

    attacks: list[dict] = []
    for frame in onset_frames:
        post_end = min(frame + 5, pitch_db.shape[1])
        if post_end <= frame:
            continue
        post = np.max(pitch_db[:, frame:post_end], axis=1)
        if frame > 3:
            pre = np.median(pitch_db[:, max(0, frame - 8):max(1, frame - 2)], axis=1)
        else:
            pre = np.full(n_semitones, -80.0, dtype=float)
        gain = post - pre
        strongest = float(np.max(post))

        local_peaks, _ = find_peaks(post, prominence=2.0, distance=2)
        candidates: list[dict] = []
        for idx in local_peaks:
            midi = int(idx + piano_min)
            if midi < min_midi or midi > max_midi:
                continue
            if post[idx] <= -48.0 or gain[idx] <= 2.5 or post[idx] <= strongest - 24.0:
                continue
            score = float(post[idx] + 0.60 * min(float(gain[idx]), 20.0))
            candidates.append(
                {
                    "midi": midi,
                    "idx": int(idx),
                    "peak_db": float(post[idx]),
                    "gain_db": float(gain[idx]),
                    "score": score,
                }
            )

        if not candidates:
            continue

        by_midi = {c["midi"]: c for c in candidates}
        kept: list[dict] = []
        for c in candidates:
            midi, amp, gain_db = c["midi"], c["peak_db"], c["gain_db"]
            alias = False
            # Strong lower fundamentals often generate these upper partials.
            for interval, margin in ((19, 4.0), (24, 4.0), (28, 4.0), (31, 5.0), (36, 5.0)):
                lower = by_midi.get(midi - interval)
                if lower and lower["peak_db"] >= amp + margin:
                    alias = True
                    break
            if not alias:
                lower_oct = by_midi.get(midi - 12)
                if (
                    lower_oct
                    and lower_oct["peak_db"] >= amp + 9.0
                    and gain_db < lower_oct["gain_db"] + 8.0
                ):
                    alias = True
            if not alias:
                kept.append(c)

        if not kept:
            continue
        kept.sort(key=lambda x: x["score"], reverse=True)
        top_score = kept[0]["score"]
        kept = [c for c in kept if c["score"] >= top_score - 18.0][:max_notes_per_attack]
        attack_time = float(librosa.frames_to_time(frame, sr=sr, hop_length=hop_length))
        for c in kept:
            c = dict(c)
            c["frame"] = int(frame)
            c["time"] = attack_time
            c["strongest_db"] = strongest
            attacks.append(c)

    if not attacks:
        return []

    # Same-pitch retriggers establish a hard maximum note release.
    same_pitch_frames: dict[int, list[int]] = defaultdict(list)
    for a in attacks:
        same_pitch_frames[a["midi"]].append(a["frame"])

    frame_times = librosa.frames_to_time(
        np.arange(pitch_db.shape[1]), sr=sr, hop_length=hop_length
    )
    max_frames = int(round(3.0 * sr / hop_length))
    min_frames = max(2, int(round(0.10 * sr / hop_length)))
    notes: list[NoteEvent] = []

    for a in attacks:
        frame = a["frame"]
        idx = a["idx"]
        pitch = a["midi"]
        peak = a["peak_db"]
        candidates_after = [f for f in same_pitch_frames[pitch] if f > frame + min_frames]
        next_same = min(candidates_after) if candidates_after else pitch_db.shape[1] - 1
        search_end = min(pitch_db.shape[1] - 1, frame + max_frames, next_same)

        # A key release normally produces a sustained run below the local peak.
        threshold = max(-53.0, peak - 19.0)
        below_run = 0
        release_frame = search_end
        for f in range(min(frame + min_frames, search_end), search_end):
            if pitch_db[idx, f] < threshold:
                below_run += 1
                if below_run >= 4:
                    release_frame = max(frame + min_frames, f - 3)
                    break
            else:
                below_run = 0

        if next_same < release_frame:
            release_frame = max(frame + min_frames, next_same - 1)

        start_sec = float(a["time"])
        end_sec = float(frame_times[min(release_frame, len(frame_times) - 1)])
        if end_sec <= start_sec:
            end_sec = start_sec + 0.10
        gain_conf = np.clip((a["gain_db"] - 2.5) / 17.5, 0.0, 1.0)
        amp_conf = np.clip((a["peak_db"] + 48.0) / 36.0, 0.0, 1.0)
        confidence = float(0.55 * gain_conf + 0.45 * amp_conf)
        notes.append(
            NoteEvent(
                start_sec=start_sec,
                end_sec=end_sec,
                midi_pitch=pitch,
                confidence=confidence,
                source="piano-poly-spectral",
                velocity=_midi_velocity(a["peak_db"], a["strongest_db"]),
            )
        )

    # De-duplicate near-identical detections of one key.
    notes.sort(key=lambda n: (n.midi_pitch, n.start_sec, -n.confidence))
    deduped: list[NoteEvent] = []
    for n in notes:
        if deduped and deduped[-1].midi_pitch == n.midi_pitch:
            prev = deduped[-1]
            if abs(n.start_sec - prev.start_sec) < 0.055:
                if n.confidence > prev.confidence:
                    deduped[-1] = n
                continue
        deduped.append(n)
    return sorted(deduped, key=lambda n: (n.start_sec, n.midi_pitch))


def assign_piano_hands(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Assign detected piano notes to a practical left/right staff.

    This is a notation heuristic rather than anatomical fingering inference.
    It favors a split near middle C, but looks for a larger gap inside each
    simultaneous attack and uses recent hand register for ambiguous single
    notes. Hand crossings remain possible.
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.start_sec, n.midi_pitch))
    groups: list[list[NoteEvent]] = []
    for n in notes:
        if not groups or n.start_sec - groups[-1][0].start_sec > 0.075:
            groups.append([n])
        else:
            groups[-1].append(n)

    last_right: tuple[float, float] | None = None
    last_left: tuple[float, float] | None = None
    for group in groups:
        pitches = sorted(n.midi_pitch for n in group)
        t = group[0].start_sec
        if len(pitches) >= 2 and pitches[0] < 62 < pitches[-1]:
            choices = []
            for lo, hi in zip(pitches, pitches[1:]):
                gap = hi - lo
                mid = (lo + hi) / 2.0
                if 50 <= mid <= 70 and gap >= 3:
                    choices.append((gap - 0.50 * abs(mid - 60.0), mid))
            split = max(choices)[1] if choices else 60.0
        else:
            split = 60.0

        for n in group:
            p = n.midi_pitch
            if len(group) > 1:
                n.hand = "left" if p < split else "right"
            elif p <= 54:
                n.hand = "left"
            elif p >= 65:
                n.hand = "right"
            else:
                # Middle-register single notes can belong to either hand.
                dl = 99.0
                dr = 99.0
                if last_left and t - last_left[1] < 2.5:
                    dl = abs(p - last_left[0])
                if last_right and t - last_right[1] < 2.5:
                    dr = abs(p - last_right[0])
                if dr + 1.5 < dl:
                    n.hand = "right"
                elif dl + 1.5 < dr:
                    n.hand = "left"
                else:
                    n.hand = "right" if p >= 60 else "left"

        left_p = [n.midi_pitch for n in group if n.hand == "left"]
        right_p = [n.midi_pitch for n in group if n.hand == "right"]
        if left_p:
            last_left = (float(np.median(left_p)), t)
        if right_p:
            last_right = (float(np.median(right_p)), t)
    return notes


def transcribe_piano_polyphonic(
    audio_path: str | Path,
    y: np.ndarray,
    sr: int,
    backend: str = "auto",
    device: str = "cpu",
) -> tuple[list[NoteEvent], str]:
    if backend not in {"auto", "transkun", "spectral"}:
        raise ValueError(f"Unsupported piano backend: {backend}")
    if backend in {"auto", "transkun"}:
        try:
            notes = transcribe_with_transkun(audio_path, device=device)
            return assign_piano_hands(notes), "transkun"
        except Exception as exc:
            if backend == "transkun":
                raise
            log.info("Transkun unavailable; using spectral piano fallback: %s", exc)
    notes = transcribe_piano_polyphonic_spectral(y, sr)
    return assign_piano_hands(notes), "piano-poly-spectral"


# v0.2 compatibility: retain a simple top-line extractor by selecting the
# highest-confidence/highest-register note near each onset from the polyphonic
# fallback. Existing callers can still import this function.
def transcribe_piano_topline(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    min_midi: int = 55,
    max_midi: int = 88,
) -> list[NoteEvent]:
    poly = transcribe_piano_polyphonic_spectral(
        y, sr, hop_length=hop_length, min_midi=min_midi, max_midi=max_midi
    )
    groups: list[list[NoteEvent]] = []
    for n in poly:
        if not groups or n.start_sec - groups[-1][0].start_sec > 0.075:
            groups.append([n])
        else:
            groups[-1].append(n)
    out = []
    for g in groups:
        plausible = [n for n in g if n.midi_pitch >= min_midi]
        if plausible:
            out.append(max(plausible, key=lambda n: (n.confidence + 0.01 * n.midi_pitch)))
    return out
