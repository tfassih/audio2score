from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from .audio import load_audio
from .piano import assign_piano_hands, read_midi_notes
from .rhythm import analyze_rhythm
from .tonal import chroma_features, detect_chords_barwise, estimate_key
from .validation import validate_piano_transcription, write_validation_artifacts
from .export import write_performance_midi
from .music import pc_name, key_fifths


def compare_audio_midi(
    audio_path: str | Path,
    midi_path: str | Path,
    output_dir: str | Path,
    *,
    strength: str = "balanced",
    add_missing: bool = True,
    substitute_pitches: bool = True,
) -> dict:
    audio_path = Path(audio_path).resolve()
    midi_path = Path(midi_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    if not midi_path.exists():
        raise FileNotFoundError(midi_path)

    y, sr = load_audio(audio_path, sr=22050)
    rhythm_y, rhythm_sr = load_audio(audio_path, sr=11025)
    rhythm = analyze_rhythm(rhythm_y, rhythm_sr)
    chroma = chroma_features(y, sr, fast=True)
    pre = detect_chords_barwise(chroma, sr, rhythm.beat_times, beats_per_bar=4, key=None)
    key = estimate_key(chroma, first_chord=pre[0] if pre else None)
    chords = detect_chords_barwise(chroma, sr, rhythm.beat_times, beats_per_bar=4, key=key)

    raw = assign_piano_hands(read_midi_notes(midi_path, source="comparison-midi"))
    result = validate_piano_transcription(
        y, sr, raw, rhythm.beat_times,
        key=key, chords=chords, strength=strength,
        add_missing=add_missing, substitute_pitches=substitute_pitches,
    )
    result.notes = assign_piano_hands(result.notes)

    base = audio_path.stem
    raw_copy = output_dir / f"{base}.01-raw-input.mid"
    shutil.copy2(midi_path, raw_copy)
    corrected_midi = output_dir / f"{base}.02-validated-performance.mid"
    write_performance_midi(corrected_midi, result.notes, rhythm.tempo_bpm, source_midi=midi_path)
    artifacts = write_validation_artifacts(result, output_dir, base)

    prefer_flats = key_fifths(key.tonic_pc, key.mode) < 0
    summary = {
        "version": "0.6.0",
        "audio": str(audio_path),
        "midi": str(midi_path),
        "tempo_bpm": rhythm.tempo_bpm,
        "key": f"{pc_name(key.tonic_pc, prefer_flats)} {key.mode}",
        "validation_strength": strength,
        "alignment": result.alignment.to_dict(),
        "validation": result.summary.to_dict(),
        "outputs": {
            "raw_input_midi": str(raw_copy),
            "validated_performance_midi": str(corrected_midi),
            **artifacts,
        },
    }
    summary_json = output_dir / f"{base}.comparison-summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["outputs"]["summary_json"] = str(summary_json)

    s = result.summary
    def pct(x):
        return "n/a" if x is None else f"{100*x:.1f}%"
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Audio2Score v0.6 comparison</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 18px}}table{{border-collapse:collapse}}td,th{{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eee;padding:2px 5px}}</style></head>
<body><h1>Audio2Score v0.6 FLAC/MIDI comparison</h1>
<p><b>Audio:</b> {audio_path.name}<br><b>MIDI:</b> {midi_path.name}</p>
<table>
<tr><th>Metric</th><th>Result</th></tr>
<tr><td>Raw MIDI notes</td><td>{s.raw_notes}</td></tr>
<tr><td>Validated notes</td><td>{s.retained_notes}</td></tr>
<tr><td>Rejected</td><td>{s.rejected_notes}</td></tr>
<tr><td>Pitch corrections</td><td>{s.corrected_pitches}</td></tr>
<tr><td>Added missing-note hypotheses</td><td>{s.added_missing_notes}</td></tr>
<tr><td>Strong support</td><td>{s.strong_support}</td></tr>
<tr><td>Ambiguous support</td><td>{s.ambiguous_support}</td></tr>
<tr><td>Weak support</td><td>{s.weak_support}</td></tr>
<tr><td>Median onset error</td><td>{s.median_onset_error_ms:.1f} ms</td></tr>
<tr><td>90th-percentile onset error</td><td>{s.p90_onset_error_ms:.1f} ms</td></tr>
<tr><td>RH weak rate</td><td>{pct(s.right_weak_rate)}</td></tr>
<tr><td>LH weak rate</td><td>{pct(s.left_weak_rate)}</td></tr>
</table>
<h2>Alignment</h2><p>scale={result.alignment.scale:.7f}, offset={result.alignment.offset_sec:+.4f}s</p>
<h2>Files</h2><p>The CSV and JSON files beside this report contain note-level evidence, missing-note candidates, and pitch substitutions.</p>
</body></html>"""
    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    summary["outputs"]["html_report"] = str(html_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio2score-compare",
        description="Compare piano MIDI hypotheses against the original audio and produce v0.6 validation diagnostics.",
    )
    p.add_argument("audio")
    p.add_argument("midi")
    p.add_argument("-o", "--output-dir", default="audio2score_comparison")
    p.add_argument("--validation-strength", choices=["conservative", "balanced", "aggressive"], default="balanced")
    p.add_argument("--no-add-missing", action="store_true")
    p.add_argument("--no-pitch-correction", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_audio_midi(
            args.audio, args.midi, args.output_dir,
            strength=args.validation_strength,
            add_missing=not args.no_add_missing,
            substitute_pitches=not args.no_pitch_correction,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
