from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Sequence

from . import asr, ingest, keyframes, media, ocr
from .artifacts import step_dir, variant
from .errors import DependencyError
from .io import read_json, write_json
from .models import Frame
from .pipeline import VideoPipeline
from .report import ReportOptions, export_report
from .settings import AppSettings, RetrySettings
from .summarize import SUMMARY_PROMPT_VERSION

Progress = Callable[[str], None]
Sleeper = Callable[[float], None]


def _existing_video(run_dir: Path) -> Path:
    extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
    video = next(
        (path for path in sorted((run_dir / "source").glob("source.*")) if path.suffix.lower() in extensions),
        None,
    )
    if not video:
        raise FileNotFoundError(f"运行目录中找不到 source 视频: {run_dir}")
    return video


def _resolve_run_dir(source: str, runs_dir: Path) -> Path | None:
    candidate = Path(source)
    if candidate.is_dir():
        return candidate.resolve()
    configured_candidate = runs_dir / candidate
    if configured_candidate.is_dir():
        return configured_candidate.resolve()
    return None


@dataclass(slots=True)
class StageContext:
    """Artifacts shared by the ordered pipeline stages."""

    source: str
    settings: AppSettings
    progress: Progress = print
    run_dir: Path | None = None
    video: Path | None = None
    audio_path: Path | None = None
    frames_dir: Path | None = None
    frames_path: Path | None = None
    ocr_path: Path | None = None
    asr_dir: Path | None = None
    asr_path: Path | None = None
    analysis_dir: Path | None = None

    def require(self, name: str) -> Path:
        value = getattr(self, name)
        if not isinstance(value, Path):
            raise RuntimeError(f"阶段上下文缺少 {name}")
        return value


class PipelineStage(ABC):
    """A retrying stage; subclasses only implement one idempotent attempt."""

    number: int
    title: str

    def __init__(self, retry: RetrySettings, sleeper: Sleeper = sleep):
        self.retry = retry
        self._sleep = sleeper

    def run(self, context: StageContext) -> None:
        started_at = perf_counter()
        for attempt in range(1, self.retry.attempts + 1):
            try:
                self._run(context)
                context.progress(
                    f"      [{self.number}/7] 完成，耗时 {perf_counter() - started_at:.2f} 秒"
                )
                return
            except Exception as exc:
                if attempt >= self.retry.attempts or not self.is_retryable(exc):
                    raise
                delay = min(
                    self.retry.initial_delay * self.retry.multiplier ** (attempt - 1),
                    self.retry.max_delay,
                )
                context.progress(
                    f"      [{self.number}/7] 第 {attempt} 次尝试失败，"
                    f"{delay:g} 秒后重试: {exc}"
                )
                self._sleep(delay)

    def is_retryable(self, exc: Exception) -> bool:
        return not isinstance(exc, (ValueError, FileNotFoundError, FileExistsError, DependencyError))

    @abstractmethod
    def _run(self, context: StageContext) -> None:
        raise NotImplementedError


class DownloadStage(PipelineStage):
    number, title = 1, "下载视频、字幕和元数据"

    def _run(self, context: StageContext) -> None:
        existing = _resolve_run_dir(context.source, context.settings.runs_dir)
        if existing:
            context.run_dir = existing
            context.video = _existing_video(existing)
            context.progress("[1/7] 复用已有下载目录")
        else:
            context.progress("[1/7] 下载视频、字幕和元数据")
            context.run_dir, context.video, _ = ingest.download_to_run(
                context.source, context.settings.runs_dir
            )
        context.progress(f"      运行目录: {context.run_dir}")


class AudioStage(PipelineStage):
    number, title = 2, "提取 ASR 音频"

    def _run(self, context: StageContext) -> None:
        audio_path = context.require("run_dir") / "audio" / "audio.wav"
        context.audio_path = audio_path
        if audio_path.exists():
            context.progress("[2/7] 复用已有音频")
            return
        context.progress("[2/7] 提取 ASR 音频")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = audio_path.with_name(f"{audio_path.stem}.tmp{audio_path.suffix}")
        temporary.unlink(missing_ok=True)
        media.extract_audio(context.require("video"), temporary)
        temporary.replace(audio_path)


class KeyframesStage(PipelineStage):
    number, title = 3, "提取关键帧"

    def _run(self, context: StageContext) -> None:
        settings = context.settings.keyframes
        frames_dir = step_dir(
            context.require("run_dir"), "keyframes",
            f"threshold-{settings.threshold}-max-{settings.max_frames}", asdict(settings),
        )
        frames_path = frames_dir / "keyframes.json"
        context.frames_dir, context.frames_path = frames_dir, frames_path
        if frames_path.exists():
            context.progress("[3/7] 复用已有关键帧")
            return
        context.progress("[3/7] 提取关键帧")
        frames = keyframes.extract_keyframes(
            context.require("video"), frames_dir, settings.threshold, settings.max_frames,
        )
        write_json(frames_path, [asdict(frame) for frame in frames])


class OCRStage(PipelineStage):
    number, title = 4, "OCR"

    def _run(self, context: StageContext) -> None:
        settings = context.settings.ocr
        config = {**asdict(settings), "keyframes": context.require("frames_dir").name}
        output_dir = step_dir(
            context.require("run_dir"), "ocr", f"{settings.engine}-{settings.language}", config,
        )
        context.ocr_path = output_dir / "frames.json"
        if context.ocr_path.exists():
            context.progress("[4/7] 复用已有 OCR")
            return
        context.progress(f"[4/7] OCR（{settings.engine}）")
        frames = [Frame(**item) for item in read_json(context.require("frames_path"))]
        recognized = ocr.recognize(
            frames, ocr.OCROptions(language=settings.language), engine=settings.engine,
        )
        write_json(context.ocr_path, [asdict(frame) for frame in recognized])


class ASRStage(PipelineStage):
    number, title = 5, "ASR"

    def _run(self, context: StageContext) -> None:
        settings = context.settings.asr
        output_dir = step_dir(
            context.require("run_dir"), "asr", f"{settings.engine}-{settings.model}", asdict(settings),
        )
        output_path = output_dir / "transcript.json"
        context.asr_dir, context.asr_path = output_dir, output_path
        if output_path.exists():
            context.progress("[5/7] 复用已有 ASR")
            return
        context.progress(f"[5/7] ASR（{settings.engine}/{settings.model}）")
        transcript = asr.transcribe(
            context.require("audio_path"),
            asr.ASROptions(
                model=settings.model, device=settings.device, compute_type=settings.compute_type,
                language=settings.language, initial_prompt=settings.initial_prompt,
                hotwords=settings.hotwords,
            ),
            engine=settings.engine,
        )
        write_json(output_path, asdict(transcript))


class AnalysisStage(PipelineStage):
    number, title = 6, "构建时间轴并生成总结"

    def _run(self, context: StageContext) -> None:
        context.progress("[6/7] 构建时间轴并生成总结")
        settings = context.settings
        config = {
            "services": settings.services, "analysis": settings.analysis,
            "asr": context.require("asr_dir").name,
            "ocr": context.require("ocr_path").parent.name,
            "summary_prompt": SUMMARY_PROMPT_VERSION,
        }
        llm = settings.services.llm
        context.analysis_dir = context.require("run_dir") / "analysis" / variant(
            f"{llm.provider}-{llm.model}", config,
        )
        # Recreate remote clients on every attempt; on-disk checkpoints make this resumable.
        VideoPipeline(settings.services).analyze(
            context.require("run_dir"), verify=settings.analysis.verify,
            vision=settings.analysis.vision, summarize=settings.analysis.summarize,
            transcript_path=context.require("asr_path"), frames_path=context.require("ocr_path"),
            analysis_dir=context.analysis_dir,
        )


class ReportStage(PipelineStage):
    number, title = 7, "导出报告"

    def _run(self, context: StageContext) -> None:
        settings = context.settings
        if not settings.report.enabled:
            context.progress("[7/7] 配置已关闭报告导出")
            return
        context.progress("[7/7] 导出报告")
        analysis_dir = context.require("analysis_dir")
        config = {"analysis": analysis_dir.name, **asdict(settings.report)}
        report_dir = context.require("run_dir") / "reports" / variant(analysis_dir.name, config)
        report_path = report_dir / settings.report.filename
        if report_path.exists():
            context.progress("      复用已有报告")
            return
        temporary = report_path.with_name(f"{report_path.stem}.tmp{report_path.suffix}")
        temporary.unlink(missing_ok=True)
        export_report(
            context.require("run_dir"), temporary,
            ReportOptions(
                include_transcript=settings.report.include_transcript,
                max_images=settings.report.max_images,
                expect_vision=settings.analysis.vision,
            ),
            analysis_dir=analysis_dir, transcript_path=context.require("asr_path"),
            frames_path=context.require("ocr_path"),
        )
        temporary.replace(report_path)


class Pipeline:
    """Run ordered stages through their single run(context) interface."""

    def __init__(self, stages: Sequence[PipelineStage]):
        self.stages = tuple(stages)

    def run(self, context: StageContext) -> Path:
        for stage in self.stages:
            stage.run(context)
        return context.require("run_dir")


def default_stages(retry: RetrySettings, sleeper: Sleeper = sleep) -> tuple[PipelineStage, ...]:
    stage_types = (
        DownloadStage, AudioStage, KeyframesStage, OCRStage, ASRStage, AnalysisStage, ReportStage,
    )
    return tuple(stage_type(retry, sleeper) for stage_type in stage_types)


def run_pipeline(
    source: str, settings: AppSettings, progress: Progress = print, *, sleeper: Sleeper = sleep,
) -> Path:
    context = StageContext(source=source, settings=settings, progress=progress)
    return Pipeline(default_stages(settings.retry, sleeper)).run(context)
