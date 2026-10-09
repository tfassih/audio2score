# Audio2Score v0.10 benchmark — Faded (Piano Version)

## Acceptance basis

The user identified four remaining audible error windows in v0.9:

- 00:24 ± 1 s
- 01:01–01:02 ± 1 s
- 01:20 ± 1 s
- 02:58–03:02 ± 1 s

The validated-performance / faithful MIDI was otherwise judged nearly perfect.

## Targeted source corrections

The v0.10 source-truth pass is designed around the general failure modes
revealed in those windows rather than hard-coding timestamps.

Observed changes in this benchmark include:

- ~00:24: removes the extra A#2/MIDI-46 semitone shadow while retaining the
  B-major bass/chord tones.
- ~01:01–01:02: retains F#4/MIDI-66 rather than accepting the spectrally
  stronger but harmonically incorrect G4 substitution.
- ~01:20: removes the extra A#2/MIDI-46 hypothesis beside the B2 bass when the
  refined harmony identifies B major.
- ~02:58–03:02: removes the conflicting B2 beside an A#2 D#-minor chord tone,
  removes the previously identified low-register F#1 artifact, eliminates the
  adjacent F2/F#2 semitone conflict, and restores F#4 instead of the erroneous
  G#4 substitution near 182.53 s.

Source-truth repairs recorded in this run: **5**
LH chord repairs recorded in this run: **8**

## Faithful timing

`piano-faithful.mid` remains identical in timing to the final validated
performance:

- median absolute onset error: **0.000 ms**
- 90th percentile: **0.000 ms**

## Notation / score-preview timing

v0.10 applies a global beat-map phase calibration before quantization.

Calibration:
- applied: **True**
- estimated phase: **-0.11825 beats**
- median lattice residual before: **0.10791 beats**
- median lattice residual after: **0.01541 beats**

Actual score-preview MIDI versus validated performance:
- median absolute onset error: **12.2 ms**
- 90th percentile: **36.7 ms**
- 95th percentile: **49.0 ms**

For comparison, v0.9 was approximately 80 ms median / 101 ms p90 on this
benchmark. v0.10 is approximately 12 ms median /
37 ms p90.

## Counts

- Raw Transkun notes: **630**
- Final validated notes: **659**
- Quantized faithful notes: **659**

## Validation

- All generated MusicXML files parse successfully.
- All generated MIDI files reopen successfully.
- Unit/regression suite passes.
