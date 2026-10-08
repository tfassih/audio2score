# Changelog

## 0.5.1

Compatibility and install reliability release.

- Requires Python >=3.12 with no artificial upper bound.
- Updated core dependency ranges for modern Python 3.12-3.14 environments.
- Fixed Windows setup working-directory assumptions.
- Added automatic Python selection and `-PythonVersion` override.
- Setup recreates stale `.venv` by default and verifies the generated CLI.
- Added `audio2score-doctor`.
- Added automatic Transkun 2.0.1 compatibility patch:
  - removes runtime dependency on `pkg_resources`;
  - replaces pydub/from_mp3 audio decoding with SoundFile;
  - avoids Python 3.13+ `audioop` failure for Audio2Score-driven Transkun runs.
- Added `audioop-lts` marker for Python 3.13+ to the neural extra as a safety
  net for other pydub code paths.
- Transkun is now invoked through the active Python interpreter instead of
  relying on a PATH-resolved executable.
- Added explicit Python 3.14 TorchScript warning and actionable failure hint.
- Removed Basic Pitch from the supported Python-3.12+ melody backend list;
  non-piano lead extraction uses pYIN in this release.
- Added compatibility regression tests.

## 0.5.0

- Phrase detection and MusicXML slurs.
- Dynamic marks and crescendo/diminuendo wedges.
- Sustain pedal planning in MusicXML and MIDI CC64.
- Section/rehearsal detection.
- Pickup handling.
- Global melody tracking.
- Faithful, Intermediate, and Easy piano arrangements.
- Cached analysis rerendering.
