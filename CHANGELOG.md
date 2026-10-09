# Audio2Score v0.10.0

- Added source-truth guards after audio validation.
- Veto unsafe pitch substitutions that move a credible chord/scale tone to a harmonically inconsistent neighbor.
- Added left-hand semitone-shadow arbitration for near-simultaneous bass hypotheses.
- Run source-truth guards both before and after conservative LH chord repair.
- Added notation beat-phase calibration before adaptive quantization.
- Reduced benchmark score-preview onset error from ~80 ms median to ~12 ms median.
- Retains v0.9's exact validated-performance → piano-faithful MIDI identity.
- Added source-truth repair diagnostics and notation timing diagnostics to analysis JSON.

# Audio2Score v0.9.0

- Made validated-performance notes authoritative for the faithful layer.
- `piano-faithful.mid` now preserves validated pitch, hand, velocity, onset, and release exactly.
- Added `piano-faithful-score-preview.mid` for auditioning notation quantization separately.
- Faithful score construction no longer re-runs hand assignment or post-validation artifact pruning.
- Faithful MusicXML preserves validated pitch/hand identity while quantizing only notation positions.
- `score_midi` now points to the score-preview MIDI rather than the faithful performance MIDI.
- Retains v0.8 validation, LH chord cleanup, repeated-note recovery, and local beat mapping.

# Audio2Score v0.8.0

- Added performance-timed score MIDI using the source's locally varying beat map.
- Faithful MIDI preserves validated source key-release durations where possible.
- Added note-driven bar harmony refinement weighted toward the left hand.
- Added conservative chord-level left-hand cleanup and pitch repair.
- Added a target-pitch spectral-flux veto to reduce false repeats during sustain.
- Changed the default piano validation profile to conservative.
- Retains v0.7 repeated-note recovery, variable beat tracking, FLAC validation,
  Transkun compatibility patches, and raw/validated/score artifact separation.

# Audio2Score v0.7.0

- Replaced the metronomic v0.6 beat grid with a locally varying performance beat map.
- Added same-key re-attack recovery for repeated piano notes merged under pedal.
- Added harmonic-family pitch evidence for low-register piano notes.
- Added wider, harmony-aware left-hand pitch repair (neighbor/fifth/octave hypotheses).
- Made conservative validation less likely to delete a strong left-hand attack solely because its fundamental is weak.
- Added micro-gap repair to reduce artificial hesitation in performance MIDI.
- Tightened adaptive attack grouping so close repeated notes are less likely to collapse.
- Retains all v0.6 raw/validated/score representation separation and diagnostics.

# Changelog

## 0.6.0

Accuracy/validation release.

- Preserves the backend's raw unquantized piano transcription as `01-raw-transcription.mid`.
- Adds an audio-validation stage before quantization.
- Aligns MIDI attacks to the source recording with a narrow global scale/offset search.
- Scores note hypotheses using octave-aware CQT pitch energy, attack gain, neighboring-pitch separation, and acoustic onset evidence.
- Adds explicit harmonic/overtone probability for suspicious octave/partial detections.
- Conservatively rejects clearly unsupported notes.
- Adds conservative neighboring-pitch substitution when the source audio strongly supports a different semitone.
- Searches for high-confidence missing note attacks in the source audio.
- Writes `02-validated-performance.mid` in real seconds without score quantization.
- Separates performance timing from score timing throughout the analysis JSON.
- Adds adaptive context-aware score quantization using dynamic programming.
- Adds `--quantizer adaptive|fixed` and validation controls.
- Adds `--piano-midi-input` for testing any existing MIDI hypothesis through the full pipeline.
- Adds the `audio2score-compare` command for standalone FLAC/MIDI diagnostics.
- Writes validation HTML/JSON/CSV diagnostics.
- Rerendering now prefers cached validated unquantized notes and requantizes them for the requested score output.
- Retains the v0.5.1 Python 3.12+ and Transkun compatibility fixes.

## 0.5.1

Compatibility/reliability release.

- Python 3.12+ core support.
- Transkun compatibility patch for obsolete `pkg_resources` resource lookup.
- Transkun FLAC loader patched to use SoundFile instead of the pydub/audioop path.
- Robust Windows setup and console-launcher verification.
- `audio2score-doctor` installation diagnostics.

## 0.5.0

Engraving/performance release.

- Global melody tracking.
- Phrase/slur detection.
- Dynamics and wedges.
- Pedal markings and MIDI CC64.
- Section markers.
- Cached rerendering.
