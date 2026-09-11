from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Services


@dataclass(frozen=True, slots=True)
class DownloadSettings:
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None

    def __post_init__(self) -> None:
        if self.cookies_from_browser and self.cookies_file:
            raise ValueError("download.cookies_from_browser 和 download.cookies_file 不能同时设置")


@dataclass(frozen=True, slots=True)
class KeyframeSettings:
    threshold: float = 0.25
    max_frames: int = 500


@dataclass(frozen=True, slots=True)
class OCRSettings:
    engine: str = "rapidocr"
    language: str = "ch"


@dataclass(frozen=True, slots=True)
class ASRSettings:
    engine: str = "faster-whisper"
    model: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    verify: bool = True
    vision: bool = False
    summarize: bool = True
    authoring: bool = True


@dataclass(frozen=True, slots=True)
class AuthoringSettings:
    evidence_window_seconds: float = 120.0
    max_units_per_map: int = 4
    max_candidate_frames_per_request: int = 12
    selected_frames_per_request: int = 3

    def __post_init__(self) -> None:
        if self.evidence_window_seconds <= 0:
            raise ValueError("authoring.evidence_window_seconds 必须为正数")
        if self.max_units_per_map < 1:
            raise ValueError("authoring.max_units_per_map 必须至少为 1")
        if self.max_candidate_frames_per_request < 1:
            raise ValueError("authoring.max_candidate_frames_per_request 必须至少为 1")
        if not 1 <= self.selected_frames_per_request <= self.max_candidate_frames_per_request:
            raise ValueError("authoring.selected_frames_per_request 必须在 1 和候选帧数之间")


@dataclass(frozen=True, slots=True)
class ReportSettings:
    enabled: bool = True
    filename: str = "final-report.pdf"
    include_transcript: bool = True
    max_images: int | None = None


@dataclass(frozen=True, slots=True)
class RetrySettings:
    attempts: int = 3
    initial_delay: float = 2.0
    multiplier: float = 2.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("retry.attempts 必须至少为 1")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("retry 延迟不能为负数")
        if self.multiplier < 1:
            raise ValueError("retry.multiplier 必须至少为 1")


@dataclass(frozen=True, slots=True)
class AppSettings:
    services: Services
    runs_dir: Path = Path("runs")
    download: DownloadSettings = field(default_factory=DownloadSettings)
    keyframes: KeyframeSettings = field(default_factory=KeyframeSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    authoring: AuthoringSettings = field(default_factory=AuthoringSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)

    @classmethod
    def load(cls, path: Path) -> AppSettings:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("配置文件顶层必须是 YAML mapping")

        def section(name: str) -> dict[str, Any]:
            value = payload.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"配置项 {name} 必须是 mapping")
            return value

        return cls(
            services=Services.from_dict(section("services")),
            runs_dir=Path(payload.get("runs_dir", "runs")),
            download=DownloadSettings(**section("download")),
            keyframes=KeyframeSettings(**section("keyframes")),
            ocr=OCRSettings(**section("ocr")),
            asr=ASRSettings(**section("asr")),
            analysis=AnalysisSettings(**section("analysis")),
            authoring=AuthoringSettings(**section("authoring")),
            report=ReportSettings(**section("report")),
            retry=RetrySettings(**section("retry")),
        )
