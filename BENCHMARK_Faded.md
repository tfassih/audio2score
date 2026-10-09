# Audio2Score v0.11 benchmark — Faded (Piano Version)

## v0.10 remaining issues

The user identified two remaining left-hand note errors around **01:20** and
**02:58**, plus one score-preview timing issue around **01:55**.

Inspection showed the two note errors were the same failure mode: a strong raw
**B2** hypothesis competed with an `audio-validation-missing` **A#2** semitone
neighbor. v0.10 then used inferred chord membership to delete the raw B2 and
keep the added A#2.

## v0.11 correction

Source provenance now breaks this specific tie *before* harmonic cleanup. When a
strong raw transcription note competes with a validator-added semitone neighbor,
the raw note is preserved unless it is genuinely weak and the added note has a
decisive evidence advantage.

### Acceptance windows

At **~01:20** v0.11 contains:
- LH **B2 (MIDI 47)** at ~80.196 s
- no competing A#2 at that attack

At **~02:58** v0.11 contains:
- LH **B1 (MIDI 35)** at ~178.522 s
- LH **B2 (MIDI 47)** at ~178.525 s
- no competing A#2 at that attack

These corrections are rule-based, not timestamp hard-codes.

## Faithful performance

`02-validated-performance.mid` and `piano-faithful.mid` remain event-identical:
- exact identity: **True**
- note count: **659**

## Score-preview timing

After the v0.10 global notation-phase correction, v0.11 adds a smooth local
playback-map fit based on the quantized score's original validated attack times.
The notation stays on normal score positions; only audition timing is refined.

Whole-file score-preview timing versus validated performance:

| Metric | v0.10 | v0.11 |
|---|---:|---:|
| Matched notes | 657 | 659 |
| Median absolute onset error | 12.2 ms | **6.1 ms** |
| 90th percentile | 36.7 ms | **27.5 ms** |
| 95th percentile | 46.5 ms | **42.9 ms** |

The isolated LH A#3 near **01:55** now starts at the same timestamp in the
score-preview and validated-performance MIDI (~115.968 s).

Internal playback-fit diagnostics:
```json
{
  "applied": true,
  "median_abs_ms_before": 10.767324263024847,
  "median_abs_ms_after": 4.875283446722278,
  "p90_abs_ms_before": 32.20598129251293,
  "p90_abs_ms_after": 23.605491142276946,
  "max_abs_correction_ms": 70.00000000000006
}
```

## Validation

- Raw Transkun notes: **630**
- Final validated notes: **659**
- Faithful notes: **659**
- All generated MusicXML files parse successfully.
- All generated MIDI files reopen successfully.
- Regression suite: **28 passed in 1.92s**
