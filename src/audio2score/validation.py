from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

import librosa
import numpy as np
from scipy.signal import find_peaks

from .models import ChordEvent, KeyEstimate, NoteEvent
from .music import QUALITY_INTERVALS
from .rhythm import time_to_beat

PIANO_MIN = 21
PIANO_MAX = 108
HARMONIC_INTERVALS = (12, 19, 24, 28, 31, 36)


@dataclass
class AlignmentResult:
    scale: float = 1.0
    offset_sec: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationSummary:
    raw_notes: int
    retained_notes: int
    corrected_pitches: int
    rejected_notes: int
    added_missing_notes: int
    strong_support: int
    ambiguous_support: int
    weak_support: int
    median_onset_error_ms: float | None
    p90_onset_error_ms: float | None
    right_weak_rate: float | None
    left_weak_rate: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    notes: list[NoteEvent]
    rejected: list[NoteEvent]
    added: list[NoteEvent]
    diagnostics: list[dict]
    missing_candidates: list[dict]
    pitch_substitutions: list[dict]
    alignment: AlignmentResult
    summary: ValidationSummary


@dataclass
class ValidationConfig:
    reject_floor: float
    ambiguous_floor: float
    strong_floor: float
    add_missing_floor: float
    substitute_floor: float
    substitute_margin: float
    harmonic_reject_floor: float
    max_missing_per_onset: int


CONFIGS = {
    "conservative": ValidationConfig(
        reject_floor=0.05,
        ambiguous_floor=0.50,
        strong_floor=0.80,
        add_missing_floor=0.97,
        substitute_floor=0.92,
        substitute_margin=0.40,
        harmonic_reject_floor=0.90,
        max_missing_per_onset=1,
    ),
    "balanced": ValidationConfig(
        reject_floor=0.08,
        ambiguous_floor=0.50,
        strong_floor=0.80,
        add_missing_floor=0.94,
        substitute_floor=0.88,
        substitute_margin=0.30,
        harmonic_reject_floor=0.78,
        max_missing_per_onset=2,
    ),
    "aggressive": ValidationConfig(
        reject_floor=0.15,
        ambiguous_floor=0.55,
        strong_floor=0.82,
        add_missing_floor=0.90,
        substitute_floor=0.82,
        substitute_margin=0.22,
        harmonic_reject_floor=0.64,
        max_missing_per_onset=3,
    ),
}


def _copy_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    return [deepcopy(n) for n in notes]


def _onset_times(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        hop_length=hop_length,
        units="time",
        backtrack=True,
        delta=0.10,
        wait=1,
    )
    return np.asarray(times, dtype=float)


def align_midi_to_audio(
    y: np.ndarray,
    sr: int,
    notes: list[NoteEvent],
    *,
    hop_length: int = 512,
) -> tuple[list[NoteEvent], AlignmentResult, np.ndarray]:
    """Estimate a small global scale/offset correction from acoustic attacks.

    Transkun normally starts nearly aligned already. This search is deliberately
    narrow: it fixes encoder/decoder latency and tiny clock-rate mismatches but
    does not warp expressive local timing.
    """
    if not notes:
        return [], AlignmentResult(), np.array([], dtype=float)

    audio_onsets = _onset_times(y, sr, hop_length)
    if audio_onsets.size == 0:
        return _copy_notes(notes), AlignmentResult(), audio_onsets

    midi_onsets = np.array(sorted({round(n.start_sec, 4) for n in notes}), dtype=float)
    if midi_onsets.size > 900:
        midi_onsets = midi_onsets[:: max(1, midi_onsets.size // 900)]

    # Use nearest-onset quality, with a soft tolerance around 80 ms.
    def score(scale: float, offset: float) -> float:
        shifted = midi_onsets * scale + offset
        idx = np.searchsorted(audio_onsets, shifted)
        left = np.abs(shifted - audio_onsets[np.clip(idx - 1, 0, len(audio_onsets) - 1)])
        right = np.abs(shifted - audio_onsets[np.clip(idx, 0, len(audio_onsets) - 1)])
        d = np.minimum(left, right)
        return float(np.mean(np.exp(-0.5 * (d / 0.080) ** 2)))

    best = AlignmentResult()
    # Coarse then fine search. The scale range is intentionally tiny.
    for scale in np.arange(0.9980, 1.0021, 0.0010):
        for offset in np.arange(-0.18, 0.181, 0.020):
            s = score(float(scale), float(offset))
            if s > best.score:
                best = AlignmentResult(float(scale), float(offset), s)
    for scale in np.arange(best.scale - 0.0006, best.scale + 0.00061, 0.0002):
        for offset in np.arange(best.offset_sec - 0.025, best.offset_sec + 0.0251, 0.005):
            s = score(float(scale), float(offset))
            if s > best.score:
                best = AlignmentResult(float(scale), float(offset), s)

    aligned = _copy_notes(notes)
    for n in aligned:
        n.start_sec = max(0.0, n.start_sec * best.scale + best.offset_sec)
        n.end_sec = max(n.start_sec + 0.02, n.end_sec * best.scale + best.offset_sec)
    return aligned, best, audio_onsets


def _pitch_matrix(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    bins_per_semitone: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    bpo = 12 * bins_per_semitone
    n_semitones = PIANO_MAX - PIANO_MIN + 1
    c = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=librosa.midi_to_hz(PIANO_MIN),
            n_bins=n_semitones * bins_per_semitone,
            bins_per_octave=bpo,
        )
    )
    db = librosa.amplitude_to_db(c, ref=np.max)
    pitch_db = np.full((n_semitones, db.shape[1]), -80.0, dtype=np.float32)
    for i in range(n_semitones):
        lo = i * bins_per_semitone
        hi = min(db.shape[0], lo + bins_per_semitone)
        pitch_db[i] = np.max(db[lo:hi], axis=0)
    times = librosa.frames_to_time(np.arange(pitch_db.shape[1]), sr=sr, hop_length=hop_length)
    return pitch_db, times


def _nearest_onset_error(t: float, onset_times: np.ndarray) -> float:
    if onset_times.size == 0:
        return 9.0
    idx = int(np.searchsorted(onset_times, t))
    vals = []
    if idx < len(onset_times):
        vals.append(abs(float(onset_times[idx]) - t))
    if idx > 0:
        vals.append(abs(float(onset_times[idx - 1]) - t))
    return min(vals) if vals else 9.0


def _pitch_evidence(
    pitch_db: np.ndarray,
    frame_times: np.ndarray,
    onset_times: np.ndarray,
    midi_pitch: int,
    time_sec: float,
) -> dict:
    if midi_pitch < PIANO_MIN or midi_pitch > PIANO_MAX:
        return {
            "support": 0.0, "peak_db": -80.0, "gain_db": 0.0,
            "pitch_margin_db": -20.0, "onset_error_sec": 9.0,
            "onset_support": 0.0,
        }
    idx = midi_pitch - PIANO_MIN
    f = int(np.searchsorted(frame_times, time_sec))
    f = int(np.clip(f, 0, pitch_db.shape[1] - 1))
    post_lo = max(0, f - 1)
    post_hi = min(pitch_db.shape[1], f + 4)
    pre_lo = max(0, f - 8)
    pre_hi = max(pre_lo + 1, f - 2)
    peak = float(np.max(pitch_db[idx, post_lo:post_hi]))
    pre = float(np.median(pitch_db[idx, pre_lo:pre_hi])) if f > 2 else -80.0
    gain = peak - pre

    neighbor_idx = [j for j in (idx - 2, idx - 1, idx + 1, idx + 2) if 0 <= j < pitch_db.shape[0]]
    if neighbor_idx:
        neighbor_peak = max(float(np.max(pitch_db[j, post_lo:post_hi])) for j in neighbor_idx)
        margin = peak - neighbor_peak
    else:
        margin = 0.0

    onset_error = _nearest_onset_error(time_sec, onset_times)
    onset_support = float(math.exp(-0.5 * (onset_error / 0.085) ** 2))

    # Calibrated against the v0.5 Faded diagnostic set.  A logistic score is
    # substantially more discriminative than a weighted average because weak
    # fundamentals should remain weak even when one other feature is strong.
    # Coefficients use dB-domain peak, attack gain, neighboring-semitone margin
    # and local attack proximity.
    z = (
        4.837
        + 0.2274581 * peak
        + 0.0647046 * gain
        + 0.12378257 * margin
        + 0.10070897 * (onset_error * 10.0)
    )
    support = float(1.0 / (1.0 + math.exp(-float(np.clip(z, -30.0, 30.0)))))
    return {
        "support": support,
        "peak_db": peak,
        "gain_db": gain,
        "pitch_margin_db": margin,
        "onset_error_sec": onset_error,
        "onset_support": onset_support,
    }


def _active_lower_notes(notes: list[NoteEvent], target: NoteEvent) -> list[NoteEvent]:
    out = []
    for n in notes:
        if n is target or n.midi_pitch >= target.midi_pitch:
            continue
        if abs(n.start_sec - target.start_sec) <= 0.090 or (
            n.start_sec <= target.start_sec <= min(n.end_sec, target.start_sec + 0.10)
        ):
            out.append(n)
    return out


def _harmonic_probability(note: NoteEvent, notes: list[NoteEvent], evidence_by_id: dict[int, dict]) -> float:
    cur = evidence_by_id[id(note)]
    best = 0.0
    for lower in _active_lower_notes(notes, note):
        interval = note.midi_pitch - lower.midi_pitch
        if interval not in HARMONIC_INTERVALS:
            continue
        le = evidence_by_id.get(id(lower))
        if not le:
            continue
        amp_advantage = le["peak_db"] - cur["peak_db"]
        gain_advantage = le["gain_db"] - cur["gain_db"]
        p = 0.30
        p += 0.30 * float(np.clip((amp_advantage + 2.0) / 12.0, 0.0, 1.0))
        p += 0.25 * float(np.clip((gain_advantage + 1.0) / 12.0, 0.0, 1.0))
        if cur["onset_support"] < 0.45:
            p += 0.15
        best = max(best, p)
    return float(np.clip(best, 0.0, 1.0))


def _scale_pitch_classes(key: KeyEstimate | None) -> set[int]:
    if key is None:
        return set(range(12))
    intervals = (0, 2, 4, 5, 7, 9, 11) if key.mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    return {(key.tonic_pc + x) % 12 for x in intervals}


def _chord_tones_at(chords: list[ChordEvent], beat: float) -> set[int]:
    for c in chords:
        if c.start_beat <= beat < c.end_beat:
            return {(c.root_pc + x) % 12 for x in QUALITY_INTERVALS.get(c.quality, (0, 4, 7))}
    return set()


def _candidate_neighbor_pitch(
    note: NoteEvent,
    pitch_db: np.ndarray,
    frame_times: np.ndarray,
    onset_times: np.ndarray,
    existing: list[NoteEvent],
    cfg: ValidationConfig,
) -> tuple[int | None, dict | None]:
    cur = _pitch_evidence(pitch_db, frame_times, onset_times, note.midi_pitch, note.start_sec)
    best_pitch = None
    best_e = None
    for delta in (-2, -1, 1, 2):
        p = note.midi_pitch + delta
        if p < PIANO_MIN or p > PIANO_MAX:
            continue
        if any(abs(x.start_sec - note.start_sec) < 0.07 and x.midi_pitch == p for x in existing):
            continue
        e = _pitch_evidence(pitch_db, frame_times, onset_times, p, note.start_sec)
        if e["support"] < cfg.substitute_floor:
            continue
        if e["support"] < cur["support"] + cfg.substitute_margin:
            continue
        if best_e is None or e["support"] > best_e["support"]:
            best_pitch, best_e = p, e
    return best_pitch, best_e


def _find_missing_candidates(
    pitch_db: np.ndarray,
    frame_times: np.ndarray,
    onset_times: np.ndarray,
    notes: list[NoteEvent],
    beat_times: list[float],
    key: KeyEstimate | None,
    chords: list[ChordEvent],
    cfg: ValidationConfig,
) -> list[dict]:
    scale_pcs = _scale_pitch_classes(key)
    candidates: list[dict] = []
    if onset_times.size == 0:
        return candidates

    for t in onset_times:
        f = int(np.searchsorted(frame_times, t))
        if f < 1 or f >= pitch_db.shape[1]:
            continue
        post_lo, post_hi = max(0, f - 1), min(pitch_db.shape[1], f + 4)
        pre_lo, pre_hi = max(0, f - 8), max(1, f - 2)
        post = np.max(pitch_db[:, post_lo:post_hi], axis=1)
        pre = np.median(pitch_db[:, pre_lo:pre_hi], axis=1) if f > 2 else np.full(pitch_db.shape[0], -80.0)
        gain = post - pre
        peaks, _ = find_peaks(post, prominence=3.0, distance=2)
        beat = time_to_beat(float(t), beat_times) if len(beat_times) >= 2 else 0.0
        chord_pcs = _chord_tones_at(chords, beat)

        local: list[dict] = []
        for idx in peaks:
            pitch = int(idx + PIANO_MIN)
            if pitch < 28 or pitch > 96:
                continue
            if post[idx] < -42.0 or gain[idx] < 5.0:
                continue
            if any(abs(n.start_sec - t) <= 0.090 and abs(n.midi_pitch - pitch) == 0 for n in notes):
                continue

            ev = _pitch_evidence(pitch_db, frame_times, onset_times, pitch, float(t))
            score = ev["support"]
            if pitch % 12 in scale_pcs:
                score += 0.04
            if chord_pcs and pitch % 12 in chord_pcs:
                score += 0.08
            elif chord_pcs:
                score -= 0.04

            # Reject likely partials of nearby lower notes or stronger candidate fundamentals.
            harmonic_prob = 0.0
            lower_sources: list[tuple[int, float]] = []
            for n in notes:
                if abs(n.start_sec - t) <= 0.10 and n.midi_pitch < pitch:
                    lower_sources.append((n.midi_pitch, 1.0))
            for j in peaks:
                lp = int(j + PIANO_MIN)
                if lp < pitch and post[j] > post[idx]:
                    lower_sources.append((lp, float(post[j] - post[idx])))
            for lp, strength in lower_sources:
                if pitch - lp in HARMONIC_INTERVALS:
                    harmonic_prob = max(harmonic_prob, min(1.0, 0.58 + 0.05 * strength))
            score -= 0.30 * harmonic_prob

            local.append({
                "time_sec": float(t),
                "midi_pitch": pitch,
                "score": float(np.clip(score, 0.0, 1.0)),
                "audio_support": ev["support"],
                "peak_db": ev["peak_db"],
                "gain_db": ev["gain_db"],
                "pitch_margin_db": ev["pitch_margin_db"],
                "harmonic_probability": harmonic_prob,
                "beat": beat,
            })

        local.sort(key=lambda x: x["score"], reverse=True)
        selected = []
        for cand in local:
            if cand["score"] < cfg.add_missing_floor:
                continue
            # Avoid adding octave stacks unless both attacks are very compelling.
            if any(abs(cand["midi_pitch"] - x["midi_pitch"]) in (12, 24) for x in selected) and cand["score"] < 0.94:
                continue
            selected.append(cand)
            if len(selected) >= cfg.max_missing_per_onset:
                break
        candidates.extend(selected)
    return candidates


def validate_piano_transcription(
    y: np.ndarray,
    sr: int,
    raw_notes: list[NoteEvent],
    beat_times: list[float],
    *,
    key: KeyEstimate | None = None,
    chords: list[ChordEvent] | None = None,
    strength: str = "balanced",
    align: bool = True,
    add_missing: bool = True,
    substitute_pitches: bool = True,
    hop_length: int = 512,
) -> ValidationResult:
    """Validate an unquantized piano transcription against the original audio.

    This stage intentionally operates *before* score quantization. It treats MIDI
    notes as hypotheses, scores each one against octave-aware CQT evidence and
    acoustic attacks, rejects only clearly unsupported hypotheses, optionally
    repairs strong neighboring-pitch substitutions, and conservatively restores
    missing attacks.
    """
    if strength not in CONFIGS:
        raise ValueError(f"Unsupported validation strength: {strength}")
    cfg = CONFIGS[strength]
    chords = chords or []

    if align:
        notes, alignment, onset_times = align_midi_to_audio(y, sr, raw_notes, hop_length=hop_length)
    else:
        notes = _copy_notes(raw_notes)
        alignment = AlignmentResult()
        onset_times = _onset_times(y, sr, hop_length)

    pitch_db, frame_times = _pitch_matrix(y, sr, hop_length=hop_length)

    evidence_by_id: dict[int, dict] = {}
    for n in notes:
        evidence_by_id[id(n)] = _pitch_evidence(
            pitch_db, frame_times, onset_times, n.midi_pitch, n.start_sec
        )
    for n in notes:
        evidence_by_id[id(n)]["harmonic_probability"] = _harmonic_probability(n, notes, evidence_by_id)

    diagnostics: list[dict] = []
    retained: list[NoteEvent] = []
    rejected: list[NoteEvent] = []
    substitutions: list[dict] = []

    for n in notes:
        e = evidence_by_id[id(n)]
        support = float(e["support"])
        harmonic = float(e["harmonic_probability"])
        n.audio_support = support
        n.onset_support = float(e["onset_support"])
        n.pitch_margin = float(e["pitch_margin_db"])
        n.harmonic_probability = harmonic

        status = "strong" if support >= cfg.strong_floor else "ambiguous" if support >= cfg.ambiguous_floor else "weak"
        reason = "supported"

        # Neighbor-pitch correction is intentionally conservative.
        if substitute_pitches and support < cfg.ambiguous_floor + 0.08:
            new_pitch, new_e = _candidate_neighbor_pitch(n, pitch_db, frame_times, onset_times, notes, cfg)
            if new_pitch is not None and new_e is not None:
                old_pitch = n.midi_pitch
                n.original_pitch = old_pitch
                n.midi_pitch = int(new_pitch)
                n.audio_support = float(new_e["support"])
                n.onset_support = float(new_e["onset_support"])
                n.pitch_margin = float(new_e["pitch_margin_db"])
                status = "corrected"
                reason = f"pitch_substitution_{old_pitch}_to_{new_pitch}"
                substitutions.append({
                    "time_sec": n.start_sec,
                    "from_midi": old_pitch,
                    "to_midi": new_pitch,
                    "old_support": support,
                    "new_support": float(new_e["support"]),
                })
                support = float(new_e["support"])
                harmonic = 0.0

        reject = False
        if status != "corrected":
            if support < cfg.reject_floor:
                reject = True
                reason = "very_low_audio_support"
            elif support < cfg.ambiguous_floor and harmonic >= cfg.harmonic_reject_floor:
                reject = True
                reason = "likely_harmonic_artifact"
            elif n.midi_pitch < 48 and support < cfg.ambiguous_floor - 0.06 and n.onset_support < 0.20:
                reject = True
                reason = "weak_low_register_hypothesis"

        n.validation_status = "rejected" if reject else status
        n.validation_reason = reason
        row = {
            "start_sec": n.start_sec,
            "end_sec": n.end_sec,
            "midi_pitch": n.midi_pitch,
            "original_pitch": n.original_pitch,
            "hand": n.hand,
            "source": n.source,
            "audio_support": n.audio_support,
            "peak_db": e["peak_db"],
            "gain_db": e["gain_db"],
            "onset_support": n.onset_support,
            "onset_error_ms": 1000.0 * float(e["onset_error_sec"]),
            "pitch_margin_db": n.pitch_margin,
            "harmonic_probability": n.harmonic_probability,
            "validation_status": n.validation_status,
            "validation_reason": n.validation_reason,
        }
        diagnostics.append(row)
        if reject:
            rejected.append(n)
        else:
            retained.append(n)

    missing_candidates = _find_missing_candidates(
        pitch_db, frame_times, onset_times, retained, beat_times, key, chords, cfg
    ) if add_missing else []

    added: list[NoteEvent] = []
    if add_missing:
        for c in missing_candidates:
            # Conservative inferred key duration: stop at next strong attack or
            # 0.75 s, whichever comes first. Score quantization will infer the
            # printed duration separately.
            t = float(c["time_sec"])
            future = onset_times[onset_times > t + 0.05]
            end = min(t + 0.75, float(future[0]) if future.size else t + 0.75)
            new = NoteEvent(
                start_sec=t,
                end_sec=max(t + 0.08, end),
                midi_pitch=int(c["midi_pitch"]),
                confidence=float(c["score"]),
                source="audio-validation-missing",
                velocity=72,
                audio_support=float(c["audio_support"]),
                onset_support=1.0,
                pitch_margin=float(c["pitch_margin_db"]),
                harmonic_probability=float(c["harmonic_probability"]),
                validation_status="added",
                validation_reason="strong_audio_attack_missing_from_midi",
            )
            retained.append(new)
            added.append(new)

    # Retriggers end the previous key event, but acoustic ringing/pedal does not
    # extend the symbolic keypress. Preserve raw model release times otherwise.
    retained.sort(key=lambda n: (n.midi_pitch, n.start_sec))
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in retained:
        by_pitch.setdefault(n.midi_pitch, []).append(n)
    for seq in by_pitch.values():
        for a, b in zip(seq, seq[1:]):
            if a.end_sec > b.start_sec:
                a.end_sec = max(a.start_sec + 0.04, b.start_sec - 0.005)
    retained = sorted(retained, key=lambda n: (n.start_sec, n.midi_pitch))

    onset_errors = [float(d["onset_error_ms"]) for d in diagnostics if d["onset_error_ms"] < 1000]
    strong = sum(1 for d in diagnostics if d["validation_status"] in {"strong", "corrected"})
    weak = sum(1 for d in diagnostics if d["validation_status"] in {"weak", "rejected"})
    ambiguous = len(diagnostics) - strong - weak

    def weak_rate(hand: str) -> float | None:
        rows = [d for d in diagnostics if d["hand"] == hand]
        if not rows:
            return None
        return sum(1 for d in rows if d["validation_status"] in {"weak", "rejected"}) / len(rows)

    summary = ValidationSummary(
        raw_notes=len(raw_notes),
        retained_notes=len(retained),
        corrected_pitches=len(substitutions),
        rejected_notes=len(rejected),
        added_missing_notes=len(added),
        strong_support=strong,
        ambiguous_support=ambiguous,
        weak_support=weak,
        median_onset_error_ms=float(np.median(onset_errors)) if onset_errors else None,
        p90_onset_error_ms=float(np.percentile(onset_errors, 90)) if onset_errors else None,
        right_weak_rate=weak_rate("right"),
        left_weak_rate=weak_rate("left"),
    )
    return ValidationResult(
        notes=retained,
        rejected=rejected,
        added=added,
        diagnostics=diagnostics,
        missing_candidates=missing_candidates,
        pitch_substitutions=substitutions,
        alignment=alignment,
        summary=summary,
    )


def write_validation_artifacts(result: ValidationResult, output_dir: str | Path, base: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / f"{base}.validation-report.json"
    diagnostics_csv = output_dir / f"{base}.note-diagnostics.csv"
    missing_csv = output_dir / f"{base}.candidate-missing-notes.csv"
    substitutions_csv = output_dir / f"{base}.pitch-substitutions.csv"
    html_path = output_dir / f"{base}.validation-report.html"

    report = {
        "alignment": result.alignment.to_dict(),
        "summary": result.summary.to_dict(),
        "rejected_notes": [n.to_dict() for n in result.rejected],
        "added_notes": [n.to_dict() for n in result.added],
        "pitch_substitutions": result.pitch_substitutions,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def write_rows(path: Path, rows: list[dict]):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = sorted({k for r in rows for k in r})
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_rows(diagnostics_csv, result.diagnostics)
    write_rows(missing_csv, result.missing_candidates)
    write_rows(substitutions_csv, result.pitch_substitutions)

    summary = result.summary
    def pct(v):
        return "n/a" if v is None else f"{100.0*v:.1f}%"
    def ms(v):
        return "n/a" if v is None else f"{v:.1f} ms"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Audio2Score v0.6 validation</title>
<style>body{{font-family:system-ui,sans-serif;max-width:920px;margin:40px auto;padding:0 18px}}table{{border-collapse:collapse}}td,th{{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}}</style></head>
<body><h1>Audio2Score v0.6 validation</h1><table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Raw notes</td><td>{summary.raw_notes}</td></tr>
<tr><td>Validated notes</td><td>{summary.retained_notes}</td></tr>
<tr><td>Rejected</td><td>{summary.rejected_notes}</td></tr>
<tr><td>Pitch corrections</td><td>{summary.corrected_pitches}</td></tr>
<tr><td>Added missing-note hypotheses</td><td>{summary.added_missing_notes}</td></tr>
<tr><td>Strong / ambiguous / weak</td><td>{summary.strong_support} / {summary.ambiguous_support} / {summary.weak_support}</td></tr>
<tr><td>Median onset error</td><td>{ms(summary.median_onset_error_ms)}</td></tr>
<tr><td>90th-percentile onset error</td><td>{ms(summary.p90_onset_error_ms)}</td></tr>
<tr><td>RH weak rate</td><td>{pct(summary.right_weak_rate)}</td></tr>
<tr><td>LH weak rate</td><td>{pct(summary.left_weak_rate)}</td></tr>
</table>
<p>Alignment: scale={result.alignment.scale:.7f}, offset={result.alignment.offset_sec:+.4f}s.</p>
<p>See the adjacent CSV files for note-level evidence, missing-note candidates, and pitch substitutions.</p></body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return {
        "validation_report": str(report_json),
        "validation_html": str(html_path),
        "note_diagnostics": str(diagnostics_csv),
        "candidate_missing_notes": str(missing_csv),
        "pitch_substitutions": str(substitutions_csv),
    }
