from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import Services
from .artifacts import find_inputs
from .io import read_json, write_json, write_text
from .llm import EmbeddingClient, RerankerClient, create_chat_client
from .markdown_artifacts import load_text_chapters, write_text_chapters
from .models import (
    ContentMap, Correction, EvidenceUnit, Frame, OutlineChapter, OutlineTopic, Term,
    TimelineSegment, Transcript, VisualAnalysis,
)
from .summarize import summarize_timeline
from .terminology import _evidence_rows, discover_terms
from .timeline import build_timeline
from .transcript import from_dict
from .verification import verify_transcript
from .vision import analyze_frames, select_visual_frames, translate_visual_summaries
from .authoring import (
    AUTHORING_PROMPT_VERSION,
    build_evidence_units, content_map_from_dict, draft_chapters, map_content, plan_outline, review_coverage,
    select_requested_frames, synthesize,
)
from .settings import AuthoringSettings


def _terms(payload: list[dict]) -> list[Term]:
    return [Term(**{**item, "evidence": _evidence_rows(item.get("evidence", []))}) for item in payload]


def _timeline(payload: list[dict]) -> list[TimelineSegment]:
    return [TimelineSegment(**{**item, "visuals": [VisualAnalysis(**row) for row in item.get("visuals", [])]}) for item in payload]


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

    def analyze(
        self, run_dir: Path, *, verify: bool = False, vision: bool = False,
        summarize: bool = False, transcript_path: Path | None = None,
        frames_path: Path | None = None, analysis_dir: Path | None = None,
        authoring: bool = False, authoring_settings: AuthoringSettings | None = None,
    ) -> Path:
        analysis_dir = analysis_dir or run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        status_path = analysis_dir / "status.json"
        if status_path.exists():
            status = read_json(status_path)
            authoring_current = status.get("authoring_prompt_version") == AUTHORING_PROMPT_VERSION
            if status.get("complete") is True and (not (authoring and summarize) or authoring_current):
                return analysis_dir
        metadata = read_json(run_dir / "source" / "metadata.json")
        transcript_candidates = [transcript_path] if transcript_path else find_inputs(run_dir, "asr", "transcript.json", "asr-*.json")
        ocr_candidates = [frames_path] if frames_path else find_inputs(run_dir, "ocr", "frames.json", "ocr-*.json")
        if not transcript_candidates or not ocr_candidates:
            raise FileNotFoundError("run 目录需要至少一份 ASR 和 OCR 产物")
        if len(transcript_candidates) > 1 or len(ocr_candidates) > 1:
            raise ValueError("发现多份 ASR/OCR 产物，请显式指定 transcript_path 和 frames_path")
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
        authoring_settings = authoring_settings or AuthoringSettings()
        units = []
        content_maps = []
        outline_payload: dict = {}
        outline = []
        authoring_cache_current = False
        if authoring and summarize:
            units_path = analysis_dir / "evidence-units.json"
            maps_path = analysis_dir / "content-map.json"
            outline_path = analysis_dir / "outline.json"
            authoring_version_path = analysis_dir / "authoring-version.json"
            authoring_cache_current = (
                authoring_version_path.exists()
                and read_json(authoring_version_path).get("prompt_version") == AUTHORING_PROMPT_VERSION
            )
            units = (
                [EvidenceUnit(**row) for row in read_json(units_path)] if units_path.exists()
                else build_evidence_units(
                    transcript, frames, metadata.get("chapters"), authoring_settings.evidence_window_seconds,
                )
            )
            write_json(units_path, [asdict(row) for row in units])
            existing_maps = (
                [content_map_from_dict(row) for row in read_json(maps_path)]
                if authoring_cache_current and maps_path.exists() else []
            )

            def save_maps(rows: list[ContentMap]) -> None:
                write_json(maps_path, [asdict(row) for row in rows])
                write_json(authoring_version_path, {"prompt_version": AUTHORING_PROMPT_VERSION})

            content_maps = map_content(
                units, self.llm, self.services.llm.model, authoring_settings.max_units_per_map,
                existing_maps=existing_maps, checkpoint=save_maps,
            )
            save_maps(content_maps)
            if authoring_cache_current and outline_path.exists():
                outline_payload = read_json(outline_path)
                outline = [OutlineChapter(
                    title=str(row.get("title", "")), start_unit=str(row.get("start_unit", "")),
                    end_unit=str(row.get("end_unit", "")), teaching_goal=str(row.get("teaching_goal", "")),
                    topics=[OutlineTopic(
                        title=str(topic.get("title", "")), summary=str(topic.get("summary", "")),
                        required_items=[str(value) for value in topic.get("required_items", [])],
                        supporting_items=[str(value) for value in topic.get("supporting_items", [])],
                    ) for topic in row.get("topics", []) if isinstance(topic, dict)],
                    visual_requests=[value for value in row.get("visual_requests", []) if isinstance(value, dict)],
                ) for row in outline_payload.get("chapters", [])]
            else:
                outline_payload, outline = plan_outline(
                    units, content_maps, metadata, self.llm, self.services.llm.model,
                )
                write_json(outline_path, outline_payload)

        requested_frames = (
            select_requested_frames(
                units, content_maps, frames, authoring_settings.max_candidate_frames_per_request,
                authoring_settings.selected_frames_per_request,
            ) if vision and authoring and content_maps else []
        )
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
                selected = requested_frames or select_visual_frames(frames, transcript, max_frames=None)
                visuals = analyze_frames(selected, self.vlm, self.services.vlm.model)
                write_json(visual_path, [asdict(row) for row in visuals])

        timeline = build_timeline(transcript, frames, terms, visuals)
        write_json(analysis_dir / "timeline.json", [asdict(row) for row in timeline])
        if summarize and authoring:
            existing_chapters = load_text_chapters(analysis_dir) if authoring_cache_current else []
            save_chapters = lambda rows: write_text_chapters(analysis_dir, rows)
            chapters = draft_chapters(
                units, content_maps, outline, visuals, self.llm, self.services.llm.model,
                existing_chapters=existing_chapters, checkpoint=save_chapters,
            )
            save_chapters(chapters)
            review_path = analysis_dir / "review.json"
            review_progress_path = analysis_dir / "review-progress.json"
            if not authoring_cache_current or not review_path.exists():
                progress = (
                    read_json(review_progress_path)
                    if authoring_cache_current and review_progress_path.exists() else {"chapters": []}
                )
                save_reviews = lambda rows: write_json(review_progress_path, {"chapters": rows})
                review = review_coverage(
                    units, content_maps, outline, chapters, self.llm, self.services.llm.model,
                    existing_reviews=progress.get("chapters", []), checkpoint=save_reviews,
                )
                write_json(review_path, review)
            summary_path = analysis_dir / "report.txt"
            if not authoring_cache_current or not summary_path.exists():
                report = synthesize(chapters, outline_payload, self.llm, self.services.llm.model)
                write_text(summary_path, report + "\n")
        elif summarize:
            summary_path = analysis_dir / "report.txt"
            summary_mode_path = analysis_dir / "summary-mode.json"
            chinese_source = (transcript.language or "").casefold().startswith(("zh", "cmn", "yue"))
            expected_mode = {
                "source_language": transcript.language,
                "translation_mode": "monolingual" if chinese_source else "bilingual_zh",
            }
            summary_current = (
                summary_path.exists() and summary_mode_path.exists()
                and read_json(summary_mode_path) == expected_mode
            )
            if not summary_current:
                saved_mode = read_json(summary_mode_path) if summary_mode_path.exists() else {}
                mode_current = saved_mode == expected_mode
                existing = load_text_chapters(analysis_dir) if mode_current else []
                save_chapters = lambda rows: write_text_chapters(analysis_dir, rows)
                write_json(summary_mode_path, expected_mode)
                chapters, report = summarize_timeline(
                    timeline, self.llm, self.services.llm.model,
                    existing_chapters=existing, checkpoint=save_chapters,
                    source_language=transcript.language,
                )
                save_chapters(chapters)
                write_text(summary_path, report + "\n")
        status = {"complete": True}
        if authoring and summarize:
            status["authoring_prompt_version"] = AUTHORING_PROMPT_VERSION
        write_json(status_path, status)
        return analysis_dir
