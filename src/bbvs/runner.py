from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import asr, ingest, keyframes, media, ocr
from .artifacts import step_dir, variant
from .io import read_json, write_json
from .models import Frame
from .pipeline import VideoPipeline
from .report import ReportOptions, export_report
from .settings import AppSettings
from .summarize import SUMMARY_PROMPT_VERSION

Progress = Callable[[str], None]


def _existing_video(run_dir: Path) -> Path:
    extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
    video = next(
        (path for path in sorted((run_dir / "source").glob("source.*")) if path.suffix.lower() in extensions),
        None,
    )
    if not video:
        raise FileNotFoundError(f"运行目录中找不到 source 视频: {run_dir}")
    return video


def run_pipeline(source: str, settings: AppSettings, progress: Progress = print) -> Path:
    candidate = Path(source)
    if candidate.is_dir():
        run_dir = candidate.resolve()
        video = _existing_video(run_dir)
        progress("[1/7] 复用已有下载目录")
    else:
        progress("[1/7] 下载视频、字幕和元数据")
        run_dir, video, _ = ingest.download_to_run(source, settings.runs_dir)
    progress(f"      运行目录: {run_dir}")

    audio_dir = run_dir / "audio"
    audio_path = audio_dir / "audio.wav"
    if not audio_path.exists():
        progress("[2/7] 提取 ASR 音频")
        audio_dir.mkdir(parents=True, exist_ok=True)
        media.extract_audio(video, audio_path)
    else:
        progress("[2/7] 复用已有音频")

    frames_config = asdict(settings.keyframes)
    frames_dir = step_dir(
        run_dir, "keyframes",
        f"threshold-{settings.keyframes.threshold}-max-{settings.keyframes.max_frames}", frames_config,
    )
    frames_path = frames_dir / "keyframes.json"
    if not frames_path.exists():
        progress("[3/7] 提取关键帧")
        frames = keyframes.extract_keyframes(
            video, frames_dir, settings.keyframes.threshold, settings.keyframes.max_frames,
        )
        write_json(frames_path, [asdict(frame) for frame in frames])
    else:
        progress("[3/7] 复用已有关键帧")

    ocr_config = {**asdict(settings.ocr), "keyframes": frames_dir.name}
    ocr_dir = step_dir(run_dir, "ocr", f"{settings.ocr.engine}-{settings.ocr.language}", ocr_config)
    ocr_path = ocr_dir / "frames.json"
    if not ocr_path.exists():
        progress(f"[4/7] OCR（{settings.ocr.engine}）")
        frames = [Frame(**item) for item in read_json(frames_path)]
        recognized = ocr.recognize(
            frames, ocr.OCROptions(language=settings.ocr.language), engine=settings.ocr.engine,
        )
        write_json(ocr_path, [asdict(frame) for frame in recognized])
    else:
        progress("[4/7] 复用已有 OCR")

    asr_config = asdict(settings.asr)
    asr_dir = step_dir(run_dir, "asr", f"{settings.asr.engine}-{settings.asr.model}", asr_config)
    asr_path = asr_dir / "transcript.json"
    if not asr_path.exists():
        progress(f"[5/7] ASR（{settings.asr.engine}/{settings.asr.model}）")
        transcript = asr.transcribe(
            audio_path,
            asr.ASROptions(
                model=settings.asr.model, device=settings.asr.device,
                compute_type=settings.asr.compute_type, language=settings.asr.language,
                initial_prompt=settings.asr.initial_prompt, hotwords=settings.asr.hotwords,
            ),
            engine=settings.asr.engine,
        )
        write_json(asr_path, asdict(transcript))
    else:
        progress("[5/7] 复用已有 ASR")

    progress("[6/7] 构建时间轴并生成总结")
    analysis_config = {
        "services": settings.services, "analysis": settings.analysis,
        "asr": asr_dir.name, "ocr": ocr_dir.name,
        "summary_prompt": SUMMARY_PROMPT_VERSION,
    }
    llm = settings.services.llm
    analysis_dir = run_dir / "analysis" / variant(f"{llm.provider}-{llm.model}", analysis_config)
    VideoPipeline(settings.services).analyze(
        run_dir, verify=settings.analysis.verify, vision=settings.analysis.vision,
        summarize=settings.analysis.summarize, transcript_path=asr_path,
        frames_path=ocr_path, analysis_dir=analysis_dir,
    )

    if settings.report.enabled:
        progress("[7/7] 导出报告")
        report_config = {"analysis": analysis_dir.name, **asdict(settings.report)}
        report_dir = run_dir / "reports" / variant(analysis_dir.name, report_config)
        report_path = report_dir / settings.report.filename
        if report_path.exists():
            progress("      复用已有报告")
            return run_dir
        export_report(
            run_dir, report_path,
            ReportOptions(
                include_transcript=settings.report.include_transcript,
                max_images=settings.report.max_images,
                expect_vision=settings.analysis.vision,
            ),
            analysis_dir=analysis_dir, transcript_path=asr_path, frames_path=ocr_path,
        )
    else:
        progress("[7/7] 配置已关闭报告导出")
    return run_dir
