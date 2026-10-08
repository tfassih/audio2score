from __future__ import annotations

import math
import numpy as np

PC_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 5: -1, 10: -2, 3: -3, 8: -4}
MINOR_FIFTHS = {9: 0, 4: 1, 11: 2, 6: 3, 1: 4, 8: 5, 3: 6, 10: 7, 2: -1, 7: -2, 0: -3, 5: -4}

QUALITY_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}

QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "+",
    "sus2": "sus2",
    "sus4": "sus4",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
}


def pc_name(pc: int, prefer_flats: bool = False) -> str:
    return (PC_NAMES_FLAT if prefer_flats else PC_NAMES_SHARP)[pc % 12]


def chord_name(root_pc: int, quality: str, bass_pc: int | None = None, prefer_flats: bool = False, fifths: int | None = None) -> str:
    if fifths is None:
        root_name = pc_name(root_pc, prefer_flats)
        bass_name = pc_name(bass_pc, prefer_flats) if bass_pc is not None else None
    else:
        root_name = pc_name_for_key(root_pc, fifths)
        bass_name = pc_name_for_key(bass_pc, fifths) if bass_pc is not None else None
    name = root_name + QUALITY_SUFFIX.get(quality, quality)
    if bass_pc is not None and bass_pc % 12 != root_pc % 12:
        name += "/" + str(bass_name)
    return name


def key_fifths(tonic_pc: int, mode: str) -> int:
    table = MAJOR_FIFTHS if mode == "major" else MINOR_FIFTHS
    return table.get(tonic_pc % 12, 0)


def midi_to_pitch_components(midi_pitch: int, prefer_flats: bool = False) -> tuple[str, int, int]:
    name = pc_name(midi_pitch % 12, prefer_flats)
    octave = midi_pitch // 12 - 1
    if len(name) == 1:
        return name, 0, octave
    accidental = name[1]
    return name[0], 1 if accidental == "#" else -1, octave


def note_name_to_hz(name: str) -> float:
    # Minimal parser for CLI bounds such as C2, F#5, Bb3.
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid note name: {name}")
    letter = name[0].upper()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if letter not in base:
        raise ValueError(f"Invalid note name: {name}")
    idx = 1
    accidental = 0
    if idx < len(name) and name[idx] in ("#", "b"):
        accidental = 1 if name[idx] == "#" else -1
        idx += 1
    octave = int(name[idx:])
    midi = (octave + 1) * 12 + base[letter] + accidental
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def key_signature_pc_spellings(fifths: int) -> dict[int, tuple[str, int]]:
    """Return pitch-class spellings implied by a conventional key signature."""
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    alters = {k: 0 for k in base}
    if fifths > 0:
        for letter in ["F", "C", "G", "D", "A", "E", "B"][:fifths]:
            alters[letter] = 1
    elif fifths < 0:
        for letter in ["B", "E", "A", "D", "G", "C", "F"][:abs(fifths)]:
            alters[letter] = -1
    return {(base[l] + alters[l]) % 12: (l, alters[l]) for l in base}


def pc_spelling_for_key(pc: int, fifths: int) -> tuple[str, int]:
    mapping = key_signature_pc_spellings(fifths)
    if pc % 12 in mapping:
        return mapping[pc % 12]
    # Chromatic tones fall back toward the key signature's accidental direction.
    name = pc_name(pc, prefer_flats=fifths < 0)
    if len(name) == 1:
        return name, 0
    return name[0], 1 if name[1] == "#" else -1


def pc_name_for_key(pc: int, fifths: int) -> str:
    step, alter = pc_spelling_for_key(pc, fifths)
    return step + ("#" * alter if alter > 0 else "b" * (-alter))


def midi_to_pitch_components_for_key(midi_pitch: int, fifths: int) -> tuple[str, int, int]:
    step, alter = pc_spelling_for_key(midi_pitch % 12, fifths)
    base_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
    octave = (int(midi_pitch) - base_pc - alter) // 12 - 1
    return step, alter, octave
