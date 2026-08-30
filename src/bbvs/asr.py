from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import BBVSError, DependencyError
from .models import Transcript, TranscriptSegment


@dataclass(slots=True)
class ASROptions:
    """Engine-neutral ASR settings exposed to callers."""

    model: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None


class ASREngine(Protocol):
    name: str

    def transcribe(self, audio: Path, options: ASROptions) -> Transcript: ...


class FasterWhisperEngine:
    name = "faster-whisper"

    def transcribe(self, audio: Path, options: ASROptions) -> Transcript:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyError("缺少 faster-whisper，请运行: uv sync --extra asr-faster-whisper") from exc

        model = WhisperModel(
            options.model, device=options.device, compute_type=options.compute_type
        )
        segments_iter, info = model.transcribe(
            str(audio),
            language=options.language,
            initial_prompt=options.initial_prompt,
            hotwords=options.hotwords,
            word_timestamps=True,
            vad_filter=True,
        )
        segments: list[TranscriptSegment] = []
        for segment in segments_iter:
            words = [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in (segment.words or [])
            ]
            confidence = None
            if segment.avg_logprob is not None:
                confidence = max(0.0, min(1.0, math.exp(segment.avg_logprob)))
            segments.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=confidence,
                    words=words,
                )
            )
        return Transcript(
            language=info.language,
            duration=info.duration,
            segments=segments,
            engine=self.name,
            model=options.model,
        )


class OpenAIWhisperEngine:
    name = "openai-whisper"

    def transcribe(self, audio: Path, options: ASROptions) -> Transcript:
        try:
            import whisper
        except ImportError as exc:
            raise DependencyError("缺少 openai-whisper，请运行: uv sync --extra asr-openai-whisper") from exc

        if options.hotwords:
            raise BBVSError(
                "openai-whisper 不支持独立 hotwords；请将这些词合并到 --initial-prompt"
            )
        device = None if options.device == "auto" else options.device
        model = whisper.load_model(options.model, device=device)
        result = model.transcribe(
            str(audio),
            language=options.language,
            initial_prompt=options.initial_prompt,
            word_timestamps=True,
        )
        segments = []
        for segment in result.get("segments", []):
            words = [
                {
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "word": word.get("word", ""),
                    "probability": word.get("probability"),
                }
                for word in segment.get("words", [])
            ]
            confidence = None
            if segment.get("avg_logprob") is not None:
                confidence = max(0.0, min(1.0, math.exp(segment["avg_logprob"])))
            segments.append(
                TranscriptSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=segment["text"].strip(),
                    confidence=confidence,
                    words=words,
                )
            )
        duration = segments[-1].end if segments else None
        return Transcript(
            language=result.get("language"),
            duration=duration,
            segments=segments,
            engine=self.name,
            model=options.model,
        )


_engines: dict[str, Callable[[], ASREngine]] = {
    FasterWhisperEngine.name: FasterWhisperEngine,
    OpenAIWhisperEngine.name: OpenAIWhisperEngine,
}


def register_engine(name: str, factory: Callable[[], ASREngine], *, replace: bool = False) -> None:
    """Register an external adapter without changing pipeline code."""
    if name in _engines and not replace:
        raise ValueError(f"ASR engine 已存在: {name}")
    _engines[name] = factory


def available_engines() -> tuple[str, ...]:
    return tuple(sorted(_engines))


def create_engine(name: str) -> ASREngine:
    try:
        return _engines[name]()
    except KeyError as exc:
        raise ValueError(f"未知 ASR engine: {name}；可选: {', '.join(available_engines())}") from exc


def transcribe(audio: Path, options: ASROptions, engine: str = "faster-whisper") -> Transcript:
    return create_engine(engine).transcribe(audio, options)
