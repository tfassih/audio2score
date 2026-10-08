# Changelog

## 0.5.0

- Replaced per-attack highest-note melody selection with global dynamic-programming voice tracking.
- Added phrase detection and MusicXML slurs.
- Added audio-envelope-derived `p`, `mp`, `mf`, and `f` markings.
- Added crescendo and diminuendo wedges.
- Added suggested sustain-pedal regions to MusicXML and CC64 sustain events to MIDI.
- Added section/repetition fingerprints and rehearsal marks such as `A`, `A′`, and `B`.
- Added conservative pickup/anacrusis detection and implicit pickup-measure export support.
- Added beat-aware rhythmic duration spelling.
- Reworked Intermediate LH into a regular bass–fifth–third–fifth broken-chord pattern.
- Reworked Easy LH into a half-note bass/fifth pattern.
- Added `--no-engraving`.
- Added `--analysis-cache` to rerender from prior analysis JSON without retranscription.
- v0.5 analysis JSON now stores `quantized_notes` before arrangement/cleanup for future lossless rerenders.
- Added engraving, pedal-MIDI, phrase, and melody-continuity tests.
- Retained v0.4 chord-aware cleanup, slash-bass inference, and key-signature-aware enharmonic spelling.

## 0.4.0

- Added a post-transcription musical cleanup stage.
- Added chord-aware overtone / false-positive pruning.
- Added sustain-pedal cleanup across harmony boundaries.
- Added post-quantization left/right-hand reassignment.
- Added melody, bass and harmony role labeling.
- Added bass-inversion inference and slash-chord output.
- Added key-signature-aware enharmonic notation.
- Added Faithful, Intermediate and Easy piano arrangements from one analysis pass.
