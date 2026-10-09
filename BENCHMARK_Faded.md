# Audio2Score v0.9 benchmark — Faded (Piano Version)

## v0.8 finding

The user's v0.8 validated-performance MIDI was reported as almost completely
accurate, while the faithful MIDI sounded less accurate.

A direct diff of the uploaded v0.8 files found:
- both contained exactly 661 pitches,
- all faithful pitches could be matched one-for-one to validated pitches,
- faithful timing moved essentially every event by roughly 60–100 ms,
- 31 notes were reassigned between hands during faithful arrangement.

Therefore the remaining audible error was primarily introduced *after*
validation, not by transcription.

## v0.9 rule

The validated performance is now authoritative for the faithful MIDI.

`piano-faithful.mid` preserves:
- every validated pitch,
- validated left/right hand,
- exact real-time onset,
- exact key-release time,
- velocity,
- source pedal CC64 events.

Notation quantization is isolated to MusicXML and to the separate:
`piano-faithful-score-preview.mid`.

## Acceptance test on this FLAC

- Validated-performance notes: **661**
- Faithful MIDI notes: **661**
- Exact note/hand/onset/release/velocity identity: **True**
- Faithful median onset difference vs validated: **0.000 ms**
- Faithful 90th-percentile onset difference: **0.000 ms**

The score-preview MIDI remains intentionally quantized:
- median absolute onset difference: **79.6 ms**
- 90th-percentile onset difference: **101.0 ms**

This makes the remaining notation error measurable without contaminating the
faithful performance artifact.

## Validation

- 3 MusicXML files parsed successfully.
- 6 MIDI files reopened successfully.
- [32m[32m[1m23 passed[0m[32m in 1.76s[0m[0m
