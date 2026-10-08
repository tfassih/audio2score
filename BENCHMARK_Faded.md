# Audio2Score v0.7 benchmark — Faded (Piano Version)

Source pair:
- original FLAC supplied by user
- v0.6 raw Transkun MIDI supplied by user

Validation mode: **conservative**

## v0.7 result

- Estimated average tempo: **89.10 BPM**
- Raw notes: **630**
- Validated performance notes: **678**
- Faithful score notes: **678**
- Recovered repeated-note attacks: **36**
- Rejected raw hypotheses: **3**
- Pitch substitutions: **7**
- Added missing-note hypotheses: **15**
- Median onset error: **16.7 ms**
- 90th percentile onset error: **54.4 ms**
- RH weak-note rate: **4.5%**
- LH weak-note rate: **9.3%**

## Direct v0.6 → v0.7 performance comparison

Validated-performance MIDI:
- v0.6 notes: 637
- v0.7 notes: 678
- v0.6 same-pitch repeats under 1 s: 123
- v0.7 same-pitch repeats under 1 s: 151

Faithful score MIDI:
- v0.6 notes: 637
- v0.7 notes: 678
- v0.6 same-pitch repeats under 1 s: 132
- v0.7 same-pitch repeats under 1 s: 139

## Main changes

1. v0.7 no longer throws away the detected performance beat positions and replaces them with a constant beat period. The score mapper uses a variable beat map, then regularizes notation afterward.
2. Long pedal-held hypotheses are scanned for credible same-key re-attacks before validation/quantization.
3. Low-register evidence includes a harmonic-family score, reducing false deletion of genuine bass attacks whose fundamentals are weak.
4. Conservative left-hand correction does not promote ordinary bass notes up an octave merely because an octave partial is stronger.
5. Tiny model-release gaps are repaired before performance-MIDI export to reduce artificial hesitation.

## Caveat

This is still inference from audio rather than ground-truth MIDI captured from the pianist. Recovered attacks and bass repairs are intentionally conservative, but they remain hypotheses.
