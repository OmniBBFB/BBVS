from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import Services
from .io import read_json, write_json
from .llm import EmbeddingClient, RerankerClient, create_chat_client
from .models import Chapter, Correction, Evidence, Frame, Term, TimelineSegment, Transcript, VisualAnalysis
from .summarize import summarize_timeline
from .terminology import discover_terms
from .timeline import build_timeline
from .transcript import from_dict
from .verification import verify_transcript
from .vision import analyze_frames, select_visual_frames, translate_visual_summaries


def _terms(payload: list[dict]) -> list[Term]:
    return [Term(**{**item, "evidence": [Evidence(**row) for row in item.get("evidence", [])]}) for item in payload]


def _timeline(payload: list[dict]) -> list[TimelineSegment]:
    return [TimelineSegment(**{**item, "visuals": [VisualAnalysis(**row) for row in item.get("visuals", [])]}) for item in payload]


def _chapters(payload: list[dict]) -> list[Chapter]:
    return [Chapter(**row) for row in payload]


class VideoPipeline:
    def __init__(self, services: Services):
        self.services = services
        self.llm = create_chat_client(services.llm)
        self.vlm = (
            create_chat_client(services.vlm)
            if services.vlm else None
        )
        self.embedding = (
            EmbeddingClient(services.embedding.base_url, services.embedding.api_key, services.embedding.timeout)
            if services.embedding else None
        )
        self.reranker = (
            RerankerClient(services.reranker.base_url, services.reranker.api_key, services.reranker.timeout)
            if services.reranker else None
        )

    def analyze(self, run_dir: Path, *, verify: bool = False, vision: bool = False, summarize: bool = False) -> Path:
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        metadata = read_json(run_dir / "source" / "metadata.json")
        transcript_candidates = sorted(run_dir.glob("asr-*.json"))
        ocr_candidates = sorted(run_dir.glob("ocr-*.json"))
        if not transcript_candidates or not ocr_candidates:
            raise FileNotFoundError("run 目录需要至少一个 asr-*.json 和 ocr-*.json")
        transcript = from_dict(read_json(transcript_candidates[0]))
        frames = [Frame(**row) for row in read_json(ocr_candidates[0])]

        term_path = analysis_dir / "terminology.json"
        terms = _terms(read_json(term_path)) if term_path.exists() else discover_terms(metadata, frames, self.llm, self.services.llm.model)
        write_json(term_path, [asdict(row) for row in terms])

        if verify:
            verified_path = analysis_dir / "verified-transcript.json"
            if verified_path.exists():
                transcript = from_dict(read_json(verified_path))
            else:
                progress_path = analysis_dir / "verification-progress.json"
                progress = read_json(progress_path) if progress_path.exists() else {}
                existing_corrections = [Correction(**row) for row in progress.get("corrections", [])]

                def save_verification_progress(processed: int, rows: list[Correction]) -> None:
                    write_json(progress_path, {
                        "processed_suspects": processed,
                        "corrections": [asdict(row) for row in rows],
                    })

                transcript, corrections = verify_transcript(
                    transcript, terms, self.llm, self.services.llm.model,
                    processed_suspects=int(progress.get("processed_suspects", 0)),
                    existing_corrections=existing_corrections,
                    checkpoint=save_verification_progress,
                )
                write_json(verified_path, asdict(transcript))
                write_json(analysis_dir / "corrections.json", [asdict(row) for row in corrections])

        visuals: list[VisualAnalysis] = []
        if vision:
            if not self.vlm or not self.services.vlm:
                raise ValueError("--vision 需要配置 vlm endpoint")
            visual_path = analysis_dir / "visual-analysis.json"
            if visual_path.exists():
                visuals = [VisualAnalysis(**row) for row in read_json(visual_path)]
                if any(not row.summary_zh for row in visuals):
                    visuals = translate_visual_summaries(visuals, self.llm, self.services.llm.model)
                    write_json(visual_path, [asdict(row) for row in visuals])
            else:
                selected = select_visual_frames(frames, transcript)
                visuals = analyze_frames(selected, self.vlm, self.services.vlm.model)
                write_json(visual_path, [asdict(row) for row in visuals])

        timeline = build_timeline(transcript, frames, terms, visuals)
        write_json(analysis_dir / "timeline.json", [asdict(row) for row in timeline])
        if summarize:
            summary_path = analysis_dir / "summary.json"
            summary_mode_path = analysis_dir / "summary-mode.json"
            chinese_source = (transcript.language or "").casefold().startswith(("zh", "cmn", "yue"))
            expected_mode = {
                "source_language": transcript.language,
                "translation_mode": "monolingual" if chinese_source else "bilingual_zh",
            }
            cached_summary = read_json(summary_path) if summary_path.exists() else {}
            summary_current = all(cached_summary.get(key) == value for key, value in expected_mode.items())
            if not summary_current:
                chapters_path = analysis_dir / "chapters.json"
                saved_mode = read_json(summary_mode_path) if summary_mode_path.exists() else {}
                mode_current = saved_mode == expected_mode
                existing = _chapters(read_json(chapters_path)) if chapters_path.exists() and mode_current else []
                valid_existing = (
                    all(not row.summary_zh and not row.title_zh and not row.key_points_zh for row in existing)
                    if chinese_source else all(row.summary_zh for row in existing)
                )
                if existing and not valid_existing:
                    existing = []
                save_chapters = lambda rows: write_json(chapters_path, [asdict(row) for row in rows])
                write_json(summary_mode_path, expected_mode)
                chapters, report = summarize_timeline(
                    timeline, self.llm, self.services.llm.model,
                    existing_chapters=existing, checkpoint=save_chapters,
                    source_language=transcript.language,
                )
                save_chapters(chapters)
                write_json(summary_path, report)
        return analysis_dir
