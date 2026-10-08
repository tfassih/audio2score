from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class NoteEvent:
    start_sec: float
    end_sec: float
    midi_pitch: int
    confidence: float = 1.0
    source: str = "lead"
    start_beat: float | None = None
    end_beat: float | None = None
    velocity: int = 80
    hand: str | None = None
    role: str | None = None
    audio_support: float | None = None
    onset_support: float | None = None
    pitch_margin: float | None = None
    harmonic_probability: float | None = None
    validation_status: str | None = None
    validation_reason: str | None = None
    original_pitch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChordEvent:
    start_beat: float
    end_beat: float
    root_pc: int
    quality: str
    confidence: float = 1.0
    bass_pc: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KeyEstimate:
    tonic_pc: int
    mode: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RhythmAnalysis:
    tempo_bpm: float
    beat_times: list[float]
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicEvent:
    beat: float
    mark: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WedgeEvent:
    start_beat: float
    end_beat: float
    kind: str  # "crescendo" | "diminuendo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PedalEvent:
    start_beat: float
    end_beat: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhraseSpan:
    start_beat: float
    end_beat: float
    start_pitch: int
    end_pitch: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectionMarker:
    beat: float
    label: str
    family: str
    repeat_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngravingPlan:
    dynamics: list[DynamicEvent] = field(default_factory=list)
    wedges: list[WedgeEvent] = field(default_factory=list)
    pedals: list[PedalEvent] = field(default_factory=list)
    phrases: list[PhraseSpan] = field(default_factory=list)
    sections: list[SectionMarker] = field(default_factory=list)
    pickup_beats: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamics": [x.to_dict() for x in self.dynamics],
            "wedges": [x.to_dict() for x in self.wedges],
            "pedals": [x.to_dict() for x in self.pedals],
            "phrases": [x.to_dict() for x in self.phrases],
            "sections": [x.to_dict() for x in self.sections],
            "pickup_beats": self.pickup_beats,
        }
