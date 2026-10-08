# Audio2Score v0.6

Audio2Score converts recorded music into editable MIDI and MusicXML. v0.6 changes the solo-piano pipeline so a transcription model's MIDI is treated as a **hypothesis about the performance**, not as finished notation.

The primary v0.6 flow is:

```text
original audio
   ↓
raw Transkun/spectral MIDI                (unquantized, preserved)
   ↓
audio validation against the FLAC
   ├─ global onset alignment
   ├─ octave-aware CQT pitch evidence
   ├─ onset evidence
   ├─ neighboring-pitch discrimination
   ├─ harmonic/overtone probability
   ├─ conservative pitch substitutions
   ├─ rejection of clearly unsupported notes
   └─ conservative missing-note recovery
   ↓
validated performance MIDI                (still unquantized)
   ↓
adaptive score quantization
   ↓
Faithful / Intermediate / Easy arrangement
   ↓
MusicXML + score MIDI + optional PDF
```

The critical design rule is that **score quantization never modifies the preserved performance representation**.

## Installation on Windows

Audio2Score core requires 64-bit CPython 3.12 or newer. Python 3.12/3.13 remain the recommended choices for Transkun because PyTorch warns that TorchScript is unsupported on Python 3.14+.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -PianoNeural
.\.venv\Scripts\Activate.ps1
audio2score-doctor
```

To force a specific installed Python:

```powershell
.\setup_windows.ps1 -PianoNeural -PythonVersion 3.13
```

The setup script applies Audio2Score's compatibility patch to the published Transkun 2.0.1 package so it does not depend on obsolete `pkg_resources` or the removed Python `audioop` module for FLAC loading.

## Normal solo-piano transcription

```powershell
audio2score "D:\Music\song.flac" --piano --piano-backend transkun
```

The default v0.6 settings are:

```text
validation: balanced
score quantizer: adaptive
arrangements: faithful + intermediate + easy
```

Typical output now includes:

```text
song.01-raw-transcription.mid
song.02-validated-performance.mid
song.piano-faithful.mid
song.piano-faithful.musicxml
song.piano-intermediate.mid
song.piano-intermediate.musicxml
song.piano-easy.mid
song.piano-easy.musicxml
song.analysis.json
song.validation-report.json
song.validation-report.html
song.note-diagnostics.csv
song.candidate-missing-notes.csv
song.pitch-substitutions.csv
```

`01-raw-transcription.mid` is the untouched backend hypothesis when Transkun is used. `02-validated-performance.mid` preserves real-second performance timing after FLAC validation. The `piano-*.mid` files are score representations and are intentionally quantized.

## Validation strength

```powershell
audio2score "song.flac" --piano --validation-strength conservative
audio2score "song.flac" --piano --validation-strength balanced
audio2score "song.flac" --piano --validation-strength aggressive
```

`balanced` is the recommended default. The validator is intentionally asymmetric: deleting or changing a model note requires stronger evidence than merely flagging it as suspicious.

Disable individual correction classes when testing:

```powershell
audio2score "song.flac" --piano --no-add-missing
audio2score "song.flac" --piano --no-pitch-correction
audio2score "song.flac" --piano --no-validation
```

## Adaptive vs fixed score quantization

v0.6 no longer snaps the *performance* MIDI to a sixteenth-note grid. Quantization happens only for notation.

```powershell
audio2score "song.flac" --piano --quantizer adaptive
```

The adaptive quantizer uses dynamic programming over performed attacks. It balances timing error against notation complexity and metrical strength, so near-quarter/eighth rhythms tend to become simple notation rather than mechanically choosing the nearest finest grid point.

For regression comparison with older versions:

```powershell
audio2score "song.flac" --piano --quantizer fixed --grid 4
```

v0.6's MusicXML exporter currently emits binary note values; explicit triplet/tuplet engraving remains a future exporter feature.

## Compare any MIDI against its source audio

v0.6 installs a second command:

```powershell
audio2score-compare "song.flac" "transcription.mid" -o comparison
```

This creates:

```text
report.html
*.comparison-summary.json
*.validation-report.json
*.note-diagnostics.csv
*.candidate-missing-notes.csv
*.pitch-substitutions.csv
*.01-raw-input.mid
*.02-validated-performance.mid
```

This is useful for comparing Transkun, another transcription model, a manually edited MIDI, or an Audio2Score score MIDI against the original recording.

## Use an existing MIDI as the raw transcription

For development or model comparisons, skip Transkun entirely:

```powershell
audio2score "song.flac" `
  --piano `
  --piano-midi-input "candidate.mid"
```

Audio2Score will treat `candidate.mid` as the raw hypothesis, validate it against the FLAC, and continue through the normal v0.6 arranging/engraving pipeline.

## Cached rerendering

The v0.6 analysis JSON stores three separate symbolic layers:

- `raw_notes`
- `validated_notes` (unquantized)
- `quantized_notes`

A later rerender starts from `validated_notes` when available:

```powershell
audio2score "song.flac" `
  --analysis-cache "song.analysis.json" `
  --quantizer adaptive `
  --arrangement all
```

This allows quantization, arranging, and engraving changes without rerunning the neural transcriber or FLAC validator.

## What validation means

Audio validation is not a ground-truth oracle. Acoustic piano creates difficult ambiguity through sustain pedal, coupled strings, octave reinforcement, room resonance, and harmonic partials. v0.6 therefore records note-level evidence instead of pretending every decision is certain.

Each raw note can carry:

```text
audio_support
onset_support
pitch_margin
harmonic_probability
validation_status
validation_reason
original_pitch (when corrected)
```

The CSV/JSON diagnostics are intentionally retained so future versions can be measured against the same failure cases.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

v0.6 ships with regression tests for transcription compatibility, arrangements, MusicXML/MIDI export, validation, adaptive quantization, enharmonic spelling, and unquantized performance MIDI.

## Known Bugs
- Is still very rough, has issues with repeating notes and note accuracy when overall tonality is ambiguous to the neural backend.
- Is shit at transcribing "regular music"; best (and intended) use is taking acoustic single-instrument music for digitizing + transcribing. While still rough, it's better than starting and stopping on the keys to write notes down on manuscript paper whilst composing.