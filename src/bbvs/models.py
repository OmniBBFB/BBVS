from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MediaInfo:
    path: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool = False
    format_name: str | None = None


@dataclass(slots=True)
class Frame:
    path: str
    timestamp: float
    scene_score: float | None = None
    text: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Transcript:
    language: str | None
    duration: float | None
    segments: list[TranscriptSegment]
    engine: str
    model: str


@dataclass(slots=True)
class Evidence:
    source: str
    text: str
    timestamp: float | None = None


@dataclass(slots=True)
class Term:
    canonical_name: str
    category: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(slots=True)
class Correction:
    start: float
    end: float
    original: str
    corrected: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VisualAnalysis:
    timestamp: float
    summary: str
    visual_type: str
    summary_zh: str = ""
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TimelineSegment:
    start: float
    end: float
    speech: str
    screen_text: list[str] = field(default_factory=list)
    visuals: list[VisualAnalysis] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Chapter:
    title: str
    start: float
    end: float
    summary: str
    key_points: list[str] = field(default_factory=list)
    title_zh: str = ""
    summary_zh: str = ""
    key_points_zh: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentManifest:
    source: str
    media: MediaInfo
    audio_path: str | None = None
    frames: list[Frame] = field(default_factory=list)
    transcript_path: str | None = None


def to_dict(value: object) -> dict[str, Any]:
    return asdict(value)  # type: ignore[arg-type]


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())
