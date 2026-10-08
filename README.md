# Audio2Score v0.5

Audio2Score turns recorded music into editable MusicXML and MIDI. 

- global melodic-voice tracking 
- phrase detection and MusicXML slurs
- RMS-derived `p / mp / mf / f` dynamics
- crescendo / diminuendo wedges between sustained dynamic changes
- suggested sustain-pedal regions and MIDI CC64 pedal playback
- 8-bar section analysis with rehearsal marks and repeated-section hints (`A`, `A′`, etc.)
- conservative pickup/anacrusis detection
- key-aware rhythm spelling that exposes syncopation rather than hiding it inside awkward durations
- **Intermediate** broken-chord left-hand accompaniment
- simple half-note bass/fifth accompaniment
- analysis-cache rerendering, so new arrangements/engraving do not require retranscribing the audio
- Analysis JSON stores the pre-arrangement quantized notes for future lossless rerenders

## Windows quick start

Install 64-bit Python 3.11 and FFmpeg, extract the project, then open PowerShell
in the project directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

Create all three solo-piano arrangements:

```powershell
audio2score "D:\Music\song.flac" --piano
```

Choose one level:

```powershell
audio2score "D:\Music\song.flac" --piano --arrangement faithful
audio2score "D:\Music\song.flac" --piano --arrangement intermediate
audio2score "D:\Music\song.flac" --piano --arrangement easy
```

Force the bundled deterministic piano engine:

```powershell
audio2score "D:\Music\song.flac" --piano --piano-backend spectral
```

The optional neural piano backend remains available:

```powershell
.\setup_windows.ps1 -PianoNeural
audio2score "D:\Music\song.flac" --piano --piano-backend transkun
```

## Fast rerendering from a prior analysis

The expensive step is audio-to-note transcription. Once a song has an
`analysis.json`, A2S can rebuild arrangements, dynamics, pedal, slurs and
MusicXML without running the detector again:

```powershell
audio2score "D:\Music\song.flac" `
  --analysis-cache "D:\Music\song.analysis.json" `
  --arrangement all
```

A2S caches `quantized_notes` before the arrangement stage, so future versions
can rerender exactly from that symbolic data. 

## Piano outputs

For `song.flac`, the default piano run creates:

```text
song.piano-faithful.musicxml
song.piano-faithful.mid
song.piano-intermediate.musicxml
song.piano-intermediate.mid
song.piano-easy.musicxml
song.piano-easy.mid
song.analysis.json
```

If MuseScore 4 is installed and detected, a PDF is exported for each score.
Otherwise open the MusicXML directly in MuseScore Studio and export/print there.

### Faithful

Preserves the cleaned transcription, adds global melody tracking, phrase slurs,
performance marks, pedal suggestions, section labels and key-aware notation.

### Intermediate

Keeps a readable RH melody/harmony texture but replaces noisy LH detail with a
regular quarter-note broken-chord pattern:

```text
beat 1: bass
beat 2: fifth
beat 3: third
beat 4: fifth
```

This can contain a similar number of note events to Faithful while still being
much easier to read because the rhythm and hand pattern are regular.

### Easy

Keeps a monophonic RH melody and uses a two-event LH measure:

```text
beats 1–2: bass
beats 3–4: fifth
```

## Engraving pipeline

```text
audio / cached analysis
        |
        +-- polyphonic notes
        +-- harmony + key
        |
        +-- global melody tracking
        +-- artifact pruning
        +-- bass inversion inference
        |
        +-- Faithful / Intermediate / Easy
        |
        +-- phrase detector ----------> slurs
        +-- RMS envelope -------------> dynamics + wedges
        +-- harmony regions ----------> pedal marks + CC64
        +-- 8-bar fingerprints -------> rehearsal marks / repeat hints
        +-- bass-phase analysis ------> pickup detection
        |
        +-- grand-staff MusicXML
        `-- two-track performance MIDI
```

## Engraving controls

Disable the engraving layer while retaining the arrangements:

```powershell
audio2score "song.flac" --piano --no-engraving
```

## Important limitations

A2S's performance marks are **musically informed suggestions**, not ground
truth. Pedal is inferred from harmony changes rather than literally detected
from an acoustic sustain mechanism; dynamics come from the recording's RMS
envelope; section labels are similarity-based; and automatic phrase detection
can disagree with a human editor.

The primary output remains editable MusicXML. The goal is to move the generated
score from "machine transcription" toward a useful first engraving while
keeping every automated decision easy to change in MuseScore.
