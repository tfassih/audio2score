from __future__ import annotations

from pathlib import Path
import json
import logging
import shutil

from .audio import load_audio, separate_vocals
from .rhythm import analyze_rhythm
from .tonal import chroma_features, estimate_key, detect_chords, detect_chords_barwise
from .melody import transcribe_melody
from .piano import transcribe_piano_polyphonic, assign_piano_hands, read_midi_notes
from .quantize import (
    quantize_notes,
    quantize_polyphonic_notes,
    adaptive_quantize_polyphonic_notes,
)
from .arrange import build_arrangements
from .engraving import build_engraving_plan, plan_for_variant
from .validation import validate_piano_transcription, write_validation_artifacts
from .export import (
    write_musicxml, write_midi, write_piano_musicxml, write_piano_midi,
    write_performance_midi, export_pdf_with_musescore,
)
from .music import chord_name, pc_name, key_fifths

log = logging.getLogger(__name__)
VERSION = "0.7.0"


def _export_piano_variant(
    output_dir: Path,
    base: str,
    variant: str,
    notes,
    chords,
    key,
    tempo_bpm: float,
    meter: str,
    title: str,
    composer: str,
    make_pdf: bool,
    engraving_plan=None,
) -> dict:
    stem = f"{base}.piano-{variant}"
    xml = output_dir / f"{stem}.musicxml"
    midi = output_dir / f"{stem}.mid"
    pdf = output_dir / f"{stem}.pdf"
    display = variant.capitalize()
    write_piano_musicxml(
        xml, notes, chords, key, tempo_bpm,
        meter=meter,
        title=f"{title} — {display}",
        composer=composer,
        engraving_plan=engraving_plan,
    )
    write_piano_midi(midi, notes, tempo_bpm, engraving_plan=engraving_plan)
    result = {"musicxml": str(xml), "midi": str(midi)}
    if make_pdf:
        try:
            pdf_result = export_pdf_with_musescore(xml, pdf)
            result["pdf"] = str(pdf_result) if pdf_result else None
            if not pdf_result:
                result["pdf_note"] = "MuseScore 4 not found; open the MusicXML in MuseScore and export PDF."
        except Exception as exc:
            result["pdf"] = None
            result["pdf_error"] = str(exc)
    return result


def _note_from_dict(d: dict):
    from .models import NoteEvent
    fields = {
        "start_sec", "end_sec", "midi_pitch", "confidence", "source",
        "start_beat", "end_beat", "velocity", "hand", "role",
        "audio_support", "onset_support", "pitch_margin",
        "harmonic_probability", "validation_status", "validation_reason",
        "original_pitch",
    }
    return NoteEvent(**{k: v for k, v in d.items() if k in fields})


def _chord_from_dict(d: dict):
    from .models import ChordEvent
    fields = {"start_beat", "end_beat", "root_pc", "quality", "confidence", "bass_pc"}
    return ChordEvent(**{k: v for k, v in d.items() if k in fields})


def _quantize_piano(notes, beat_times, *, quantizer: str, grid: int):
    if quantizer == "adaptive":
        return adaptive_quantize_polyphonic_notes(
            notes, beat_times, max_subdivisions=grid
        )
    if quantizer == "fixed":
        return quantize_polyphonic_notes(
            notes, beat_times, subdivisions_per_beat=grid
        )
    raise ValueError(f"Unsupported quantizer: {quantizer}")


def rerender_from_analysis(
    input_path: str | Path,
    analysis_cache: str | Path,
    output_dir: str | Path,
    *,
    arrangement: str = "all",
    make_pdf: bool = True,
    engraving: bool = True,
    title: str | None = None,
    composer: str | None = None,
    quantizer: str = "adaptive",
    grid: int = 4,
) -> dict:
    """Rebuild v0.7 score outputs from a prior analysis without retranscribing.

    v0.6 prefers cached *validated unquantized* notes, then quantizes only for
    notation. Older caches fall back to quantized/faithful notes.
    """
    input_path = Path(input_path).resolve()
    cache_path = Path(analysis_cache).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    beat_times = [float(x) for x in cached.get("beat_times", [])]
    validated_dicts = cached.get("validated_notes") or []
    if validated_dicts and beat_times:
        source_notes = [_note_from_dict(x) for x in validated_dicts]
        quantized = _quantize_piano(source_notes, beat_times, quantizer=quantizer, grid=grid)
        source_kind = "validated-unquantized"
    else:
        source_note_dicts = cached.get("quantized_notes") or cached.get("notes") or []
        if not source_note_dicts:
            raise ValueError("Analysis cache contains no reusable piano notes")
        quantized = [_note_from_dict(x) for x in source_note_dicts]
        source_kind = "quantized-legacy"

    chords = [_chord_from_dict(x) for x in cached.get("chords", [])]
    key_data = cached.get("key") or {}
    from .models import KeyEstimate
    key = KeyEstimate(
        int(key_data.get("tonic_pc", 0)),
        str(key_data.get("mode", "major")),
        float(key_data.get("confidence", 0.0)),
    )
    tempo_bpm = float(cached.get("tempo_bpm", 120.0))
    meter = str(cached.get("meter", "4/4"))
    title = title or str(cached.get("title") or input_path.stem)
    composer = composer if composer is not None else str(cached.get("composer") or "")

    arrangements = build_arrangements(quantized, chords)
    chords = arrangements.chords
    faithful = arrangements.faithful

    full_y, sr = load_audio(input_path, sr=22050)
    engraving_plan = None
    if engraving:
        engraving_plan = build_engraving_plan(
            full_y, sr, beat_times, faithful, chords, meter
        )

    allowed = {"all", "faithful", "intermediate", "easy"}
    if arrangement not in allowed:
        raise ValueError(f"Unsupported arrangement: {arrangement}")
    variants = {
        "faithful": arrangements.faithful,
        "intermediate": arrangements.intermediate,
        "easy": arrangements.easy,
    }
    selected = variants if arrangement == "all" else {arrangement: variants[arrangement]}
    base = input_path.stem
    outputs: dict = {}
    for name, variant_notes in selected.items():
        outputs[name] = _export_piano_variant(
            output_dir, base, name, variant_notes, chords, key,
            tempo_bpm, meter, title, composer, make_pdf,
            engraving_plan=plan_for_variant(engraving_plan, variant_notes, meter),
        )
    if "faithful" in outputs:
        outputs["musicxml"] = outputs["faithful"]["musicxml"]
        outputs["midi"] = outputs["faithful"]["midi"]
        outputs["score_midi"] = outputs["faithful"]["midi"]
        if "pdf" in outputs["faithful"]:
            outputs["pdf"] = outputs["faithful"].get("pdf")

    prefer_flats = key_fifths(key.tonic_pc, key.mode) < 0
    data = {
        "version": VERSION,
        "mode": "piano-validated-arrangements",
        "input": str(input_path),
        "source_analysis_cache": str(cache_path),
        "cache_input_kind": source_kind,
        "title": title,
        "composer": composer,
        "meter": meter,
        "tempo_bpm": tempo_bpm,
        "key": {
            **key.to_dict(),
            "name": f"{pc_name(key.tonic_pc, prefer_flats)} {key.mode}",
        },
        "beat_times": beat_times,
        "backend": str(cached.get("backend", "cached")),
        "quantizer": quantizer,
        "raw_note_count": int(cached.get("raw_note_count", len(quantized))),
        "validated_note_count": int(cached.get("validated_note_count", len(validated_dicts) or len(quantized))),
        "quantized_note_count": len(quantized),
        "raw_notes": cached.get("raw_notes"),
        "validated_notes": cached.get("validated_notes"),
        "quantized_notes": [n.to_dict() for n in quantized],
        "notes": [n.to_dict() for n in faithful],
        "validation": cached.get("validation"),
        "hand_counts": {
            "right": sum(1 for n in faithful if n.hand == "right"),
            "left": sum(1 for n in faithful if n.hand == "left"),
            "unassigned": sum(1 for n in faithful if n.hand not in {"left", "right"}),
        },
        "role_counts": {
            role: sum(1 for n in faithful if n.role == role)
            for role in ("melody", "bass", "harmony", "accompaniment")
        },
        "arrangement_counts": {k: len(v) for k, v in variants.items()},
        "chords": [
            {
                **c.to_dict(),
                "symbol": chord_name(
                    c.root_pc, c.quality, c.bass_pc, prefer_flats,
                    fifths=key_fifths(key.tonic_pc, key.mode),
                ),
            }
            for c in chords
        ],
        "engraving": engraving_plan.to_dict() if engraving_plan else None,
        "arrangements": {
            "intermediate": [n.to_dict() for n in arrangements.intermediate],
            "easy": [n.to_dict() for n in arrangements.easy],
        },
        "outputs": outputs,
    }
    analysis_json = output_dir / f"{base}.analysis.json"
    analysis_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    outputs["analysis_json"] = str(analysis_json)
    return data


def transcribe_song(
    input_path: str | Path,
    output_dir: str | Path,
    title: str | None = None,
    composer: str = "",
    meter: str = "4/4",
    grid: int = 4,
    melody_backend: str = "auto",
    separate: bool = False,
    separator_preset: str = "vocal_balanced",
    fmin: str = "C2",
    fmax: str = "C7",
    make_pdf: bool = True,
    keep_stems: bool = True,
    piano: bool = False,
    piano_backend: str = "auto",
    device: str = "cpu",
    arrangement: str = "all",
    engraving: bool = True,
    validation: bool = True,
    validation_strength: str = "balanced",
    add_missing: bool = True,
    substitute_pitches: bool = True,
    quantizer: str = "adaptive",
    piano_midi_input: str | Path | None = None,
) -> dict:
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    title = title or input_path.stem
    stem_dir = output_dir / "stems"
    base = input_path.stem

    analysis_sr = 22050
    full_y, sr = load_audio(input_path, sr=analysis_sr)
    if piano:
        rhythm_y, rhythm_sr = load_audio(input_path, sr=11025)
        rhythm = analyze_rhythm(rhythm_y, rhythm_sr)
    else:
        rhythm = analyze_rhythm(full_y, sr)
    log.info("Tempo estimate: %.2f BPM", rhythm.tempo_bpm)

    melody_path = input_path
    melody_y = full_y
    harmony_y = full_y
    separation_info: dict[str, str] = {}
    if separate and not piano:
        sep = separate_vocals(input_path, stem_dir, preset=separator_preset)
        if sep.vocals:
            melody_path = sep.vocals
            melody_y, _ = load_audio(melody_path, sr=sr)
            separation_info["vocals"] = str(sep.vocals)
        if sep.instrumental:
            harmony_y, _ = load_audio(sep.instrumental, sr=sr)
            separation_info["instrumental"] = str(sep.instrumental)

    chroma = chroma_features(harmony_y, sr, fast=piano)
    engraving_plan = None
    validation_result = None
    validation_outputs: dict[str, str] = {}
    raw_midi_path: Path | None = None
    validated_midi_path: Path | None = None

    if piano:
        preliminary = detect_chords_barwise(
            chroma, sr, rhythm.beat_times, beats_per_bar=4, key=None
        )
        key = estimate_key(chroma, first_chord=preliminary[0] if preliminary else None)
        chords = detect_chords_barwise(
            chroma, sr, rhythm.beat_times, beats_per_bar=4, key=key
        )

        raw_midi_path = output_dir / f"{base}.01-raw-transcription.mid"
        if piano_midi_input:
            midi_input = Path(piano_midi_input).resolve()
            if not midi_input.exists():
                raise FileNotFoundError(midi_input)
            shutil.copy2(midi_input, raw_midi_path)
            raw_notes = assign_piano_hands(read_midi_notes(midi_input, source="external-piano-midi"))
            backend_used = "external-midi"
        else:
            raw_notes, backend_used = transcribe_piano_polyphonic(
                input_path, melody_y, sr, backend=piano_backend, device=device,
                raw_midi_path=raw_midi_path,
            )
            if not raw_midi_path.exists():
                # Spectral backend has no native MIDI file, so preserve its exact
                # unquantized NoteEvent representation as the raw artifact.
                write_performance_midi(raw_midi_path, raw_notes, rhythm.tempo_bpm)

        raw_notes = assign_piano_hands(raw_notes)

        if validation:
            validation_result = validate_piano_transcription(
                full_y, sr, raw_notes, rhythm.beat_times,
                key=key, chords=chords,
                strength=validation_strength,
                add_missing=add_missing,
                substitute_pitches=substitute_pitches,
            )
            validated_notes = assign_piano_hands(validation_result.notes)
            validation_outputs = write_validation_artifacts(validation_result, output_dir, base)
            log.info(
                "Audio validation: %d raw -> %d validated (%d rejected, %d corrected, %d added)",
                len(raw_notes), len(validated_notes),
                validation_result.summary.rejected_notes,
                validation_result.summary.corrected_pitches,
                validation_result.summary.added_missing_notes,
            )
        else:
            validated_notes = assign_piano_hands(raw_notes)

        validated_midi_path = output_dir / f"{base}.02-validated-performance.mid"
        write_performance_midi(
            validated_midi_path, validated_notes, rhythm.tempo_bpm,
            source_midi=raw_midi_path,
        )

        quantized = _quantize_piano(
            validated_notes, rhythm.beat_times, quantizer=quantizer, grid=grid
        )
        arrangements = build_arrangements(quantized, chords)
        chords = arrangements.chords
        notes = arrangements.faithful
        if engraving:
            engraving_plan = build_engraving_plan(
                full_y, sr, rhythm.beat_times, arrangements.faithful, chords, meter
            )
    else:
        key = estimate_key(chroma)
        chords = detect_chords(chroma, sr, rhythm.beat_times)
        raw_notes, backend_used = transcribe_melody(
            melody_path, melody_y, sr,
            backend=melody_backend, fmin=fmin, fmax=fmax,
        )
        validated_notes = raw_notes
        notes = quantize_notes(
            raw_notes, rhythm.beat_times, subdivisions_per_beat=grid
        )
        arrangements = None
        quantized = notes

    prefer_flats = key_fifths(key.tonic_pc, key.mode) < 0
    log.info(
        "Key estimate: %s %s",
        pc_name(key.tonic_pc, prefer_flats), key.mode,
    )
    log.info(
        "Backend: %s; %d raw -> %d primary score notes",
        backend_used, len(raw_notes), len(notes),
    )

    analysis_json = output_dir / f"{base}.analysis.json"
    outputs: dict = {}
    arrangement_counts: dict[str, int] = {}

    if piano:
        outputs["raw_transcription_midi"] = str(raw_midi_path)
        outputs["validated_performance_midi"] = str(validated_midi_path)
        outputs.update(validation_outputs)
        allowed = {"all", "faithful", "intermediate", "easy"}
        if arrangement not in allowed:
            raise ValueError(f"Unsupported arrangement: {arrangement}")
        variants = {
            "faithful": arrangements.faithful,
            "intermediate": arrangements.intermediate,
            "easy": arrangements.easy,
        }
        selected = variants if arrangement == "all" else {arrangement: variants[arrangement]}
        for name, variant_notes in selected.items():
            outputs[name] = _export_piano_variant(
                output_dir, base, name, variant_notes, chords, key,
                rhythm.tempo_bpm, meter, title, composer, make_pdf,
                engraving_plan=plan_for_variant(engraving_plan, variant_notes, meter),
            )
        if "faithful" in outputs:
            outputs["musicxml"] = outputs["faithful"]["musicxml"]
            outputs["midi"] = outputs["faithful"]["midi"]
            outputs["score_midi"] = outputs["faithful"]["midi"]
            if "pdf" in outputs["faithful"]:
                outputs["pdf"] = outputs["faithful"].get("pdf")
        arrangement_counts = {k: len(v) for k, v in variants.items()}
    else:
        musicxml = output_dir / f"{base}.lead-sheet.musicxml"
        midi = output_dir / f"{base}.lead-sheet.mid"
        pdf = output_dir / f"{base}.lead-sheet.pdf"
        write_musicxml(
            musicxml, notes, chords, key, rhythm.tempo_bpm,
            meter=meter, title=title, composer=composer,
        )
        write_midi(midi, notes, chords, rhythm.tempo_bpm, include_chords=True)
        outputs = {"musicxml": str(musicxml), "midi": str(midi)}
        if make_pdf:
            try:
                pdf_result = export_pdf_with_musescore(musicxml, pdf)
                outputs["pdf"] = str(pdf_result) if pdf_result else None
                if not pdf_result:
                    outputs["pdf_note"] = "MuseScore 4 not found; open the MusicXML in MuseScore and export PDF."
            except Exception as exc:
                outputs["pdf"] = None
                outputs["pdf_error"] = str(exc)

    validation_data = None
    if validation_result:
        validation_data = {
            "strength": validation_strength,
            "alignment": validation_result.alignment.to_dict(),
            "summary": validation_result.summary.to_dict(),
            "pitch_substitutions": validation_result.pitch_substitutions,
            "rejected_notes": [n.to_dict() for n in validation_result.rejected],
            "added_notes": [n.to_dict() for n in validation_result.added],
        }

    data = {
        "version": VERSION,
        "mode": "piano-validated-arrangements" if piano else "general",
        "input": str(input_path),
        "title": title,
        "composer": composer,
        "meter": meter,
        "tempo_bpm": rhythm.tempo_bpm,
        "key": {
            **key.to_dict(),
            "name": f"{pc_name(key.tonic_pc, prefer_flats)} {key.mode}",
        },
        "beat_times": rhythm.beat_times,
        "backend": backend_used,
        "separation": separation_info,
        "quantizer": quantizer if piano else "fixed",
        "grid": grid,
        "raw_note_count": len(raw_notes),
        "validated_note_count": len(validated_notes),
        "quantized_note_count": len(quantized),
        "raw_notes": [n.to_dict() for n in raw_notes] if piano else None,
        "validated_notes": [n.to_dict() for n in validated_notes] if piano else None,
        "quantized_notes": [n.to_dict() for n in quantized] if piano else None,
        "notes": [n.to_dict() for n in notes],
        "validation": validation_data,
        "hand_counts": {
            "right": sum(1 for n in notes if n.hand == "right"),
            "left": sum(1 for n in notes if n.hand == "left"),
            "unassigned": sum(1 for n in notes if n.hand not in {"left", "right"}),
        },
        "role_counts": {
            role: sum(1 for n in notes if n.role == role)
            for role in ("melody", "bass", "harmony", "accompaniment")
        } if piano else {},
        "arrangement_counts": arrangement_counts,
        "chords": [
            {
                **c.to_dict(),
                "symbol": chord_name(
                    c.root_pc, c.quality, c.bass_pc, prefer_flats,
                    fifths=key_fifths(key.tonic_pc, key.mode),
                ),
            }
            for c in chords
        ],
        "engraving": engraving_plan.to_dict() if engraving_plan else None,
        "outputs": outputs,
    }
    if piano:
        data["arrangements"] = {
            "intermediate": [n.to_dict() for n in arrangements.intermediate],
            "easy": [n.to_dict() for n in arrangements.easy],
        }

    analysis_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    outputs["analysis_json"] = str(analysis_json)
    return data
