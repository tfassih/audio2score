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
