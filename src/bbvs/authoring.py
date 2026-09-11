from __future__ import annotations

import json
from dataclasses import asdict
from collections.abc import Callable
from typing import Any

from .llm import ChatModel, parse_json_content
from .models import Chapter, ContentMap, EvidenceUnit, Frame, OutlineChapter, Transcript, VisualAnalysis

AUTHORING_PROMPT_VERSION = "evidence-outline-draft-v1"


def build_evidence_units(
    transcript: Transcript,
    frames: list[Frame],
    metadata_chapters: list[dict[str, Any]] | None = None,
    target_seconds: float = 120.0,
) -> list[EvidenceUnit]:
    """Build stable evidence envelopes. Their boundaries are not writing chapters."""
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    chapters = metadata_chapters or []
    duration = transcript.duration or (transcript.segments[-1].end if transcript.segments else 0.0)
    units: list[EvidenceUnit] = []
    start = 0.0
    while start < duration:
        nominal_end = min(duration, start + target_seconds)
        crossing = next((row.end for row in transcript.segments if row.start < nominal_end < row.end), None)
        end = min(duration, crossing) if crossing else nominal_end
        speech = " ".join(row.text for row in transcript.segments if row.end > start and row.start < end)
        screen_text = list(dict.fromkeys(
            text for frame in frames if start <= frame.timestamp < end for text in frame.text
        ))
        source_chapter = next((
            str(row.get("title", "")) for row in chapters
            if float(row.get("start_time", row.get("start", 0))) <= start
            < float(row.get("end_time", row.get("end", duration)))
        ), None)
        units.append(EvidenceUnit(
            unit_id=f"u_{len(units) + 1:04d}", start=start, end=end,
            speech=speech, screen_text=screen_text, source_chapter=source_chapter,
        ))
        start = end
    return units


def map_content(
    units: list[EvidenceUnit], client: ChatModel, model: str, batch_size: int = 4,
    existing_maps: list[ContentMap] | None = None,
    checkpoint: Callable[[list[ContentMap]], None] | None = None,
) -> list[ContentMap]:
    maps = list(existing_maps or [])
    completed_ids = {row.unit_id for row in maps}
    pending = [row for row in units if row.unit_id not in completed_ids]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt = (
            "Create a high-recall content map for every evidence unit. Do not summarize for brevity and do not "
            "invent facts. Preserve concrete claims, mechanisms, examples, counterexamples, limitations, formulas "
            "or code mentioned by speech/OCR, and uncertainty. Identify visual requests when seeing the original "
            "slide would materially help; formulas only need their original slide, not transcription. Return JSON "
            "with maps, one item per unit, using exactly: unit_id, teaching_goal, claims, mechanisms, examples, "
            "formulas_or_code, limitations, visual_requests, uncertainties. Each visual request has description, "
            "start, end, importance(required|useful).\n" + json.dumps([asdict(row) for row in batch], ensure_ascii=False)
        )
        payload = parse_json_content(client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=4096,
            response_format={"type": "json_object"},
        ))
        by_id = {str(row.get("unit_id")): row for row in payload.get("maps", [])}
        for unit in batch:
            row = by_id.get(unit.unit_id, {})
            maps.append(ContentMap(
                unit_id=unit.unit_id,
                teaching_goal=str(row.get("teaching_goal", "")),
                claims=[str(value) for value in row.get("claims", [])],
                mechanisms=[str(value) for value in row.get("mechanisms", [])],
                examples=[str(value) for value in row.get("examples", [])],
                formulas_or_code=[str(value) for value in row.get("formulas_or_code", [])],
                limitations=[str(value) for value in row.get("limitations", [])],
                visual_requests=[value for value in row.get("visual_requests", []) if isinstance(value, dict)],
                uncertainties=[str(value) for value in row.get("uncertainties", [])],
            ))
        if checkpoint:
            checkpoint(maps)
    return maps


def select_requested_frames(
    units: list[EvidenceUnit], maps: list[ContentMap], frames: list[Frame],
    max_candidates_per_request: int = 12, selected_per_request: int = 3,
) -> list[Frame]:
    """Select a small candidate set per semantic request, without a video-wide cap."""
    unit_by_id = {row.unit_id: row for row in units}
    selected: list[Frame] = []
    seen: set[str] = set()
    for content in maps:
        unit = unit_by_id[content.unit_id]
        requests = list(content.visual_requests)
        if content.formulas_or_code and not requests:
            requests = [{"description": "formula or code slide", "start": unit.start, "end": unit.end,
                         "importance": "required"}]
        for request in requests:
            start = float(request.get("start", unit.start)) - 8
            end = float(request.get("end", unit.end)) + 8
            candidates = [row for row in frames if start <= row.timestamp <= end]
            if not candidates and frames:
                midpoint = (unit.start + unit.end) / 2
                candidates = sorted(frames, key=lambda row: abs(row.timestamp - midpoint))[:1]
            # Text-rich and later frames are better proxies for a fully revealed PPT.
            ranked = sorted(candidates, key=lambda row: (len("".join(row.text)), row.timestamp), reverse=True)
            pool = ranked[:max_candidates_per_request]
            # Keep up to three distinct candidates for VLM/agent comparison; no global frame limit.
            for frame in sorted(pool[:selected_per_request], key=lambda row: row.timestamp):
                if frame.path not in seen:
                    seen.add(frame.path)
                    selected.append(frame)
    return sorted(selected, key=lambda row: row.timestamp)


def plan_outline(
    units: list[EvidenceUnit], maps: list[ContentMap], metadata: dict[str, Any],
    client: ChatModel, model: str,
) -> tuple[dict[str, Any], list[OutlineChapter]]:
    unit_ranges = [{"unit_id": row.unit_id, "start": row.start, "end": row.end,
                    "source_chapter": row.source_chapter} for row in units]
    prompt = (
        "Design a global teaching outline from the content maps. Reorganize by conceptual dependency rather than "
        "fixed time windows, while keeping every chapter backed by a contiguous source range. Preserve all high-value "
        "claims, examples, formulas/code and closing limitations. Keep the output bounded: concept_dependencies "
        "must be an array of at most 12 short strings; must_preserve at most 20 short strings. Return JSON with "
        "exactly central_question, thesis, concept_dependencies, must_preserve, and chapters. Each chapter uses "
        "exactly title, start_unit, end_unit, teaching_goal, required_units, visual_requests. "
        "Every unit must belong to exactly one chapter and unit ranges "
        "must be ordered, contiguous and non-overlapping.\n"
        + json.dumps({"metadata": metadata, "units": unit_ranges,
                      "content_maps": [asdict(row) for row in maps]}, ensure_ascii=False)
    )
    payload = parse_json_content(client.chat(
        model=model, messages=[{"role": "system", "content": "You are the lead editor of a rigorous course note."},
                               {"role": "user", "content": prompt}],
        max_tokens=8192, response_format={"type": "json_object"},
    ))
    chapters = [OutlineChapter(
        title=str(row.get("title", "")), start_unit=str(row.get("start_unit", "")),
        end_unit=str(row.get("end_unit", "")), teaching_goal=str(row.get("teaching_goal", "")),
        required_units=[str(value) for value in row.get("required_units", [])],
        visual_requests=[value for value in row.get("visual_requests", []) if isinstance(value, dict)],
    ) for row in payload.get("chapters", [])]
    _validate_outline(units, chapters)
    return payload, chapters


def _validate_outline(units: list[EvidenceUnit], chapters: list[OutlineChapter]) -> None:
    ids = [row.unit_id for row in units]
    positions = {value: index for index, value in enumerate(ids)}
    covered: list[str] = []
    for chapter in chapters:
        if chapter.start_unit not in positions or chapter.end_unit not in positions:
            raise ValueError("outline references an unknown evidence unit")
        left, right = positions[chapter.start_unit], positions[chapter.end_unit]
        if left > right:
            raise ValueError("outline chapter range is reversed")
        covered.extend(ids[left:right + 1])
    if covered != ids:
        raise ValueError("outline chapters must cover all evidence units exactly once in source order")


def draft_chapters(
    units: list[EvidenceUnit], maps: list[ContentMap], outline: list[OutlineChapter],
    visuals: list[VisualAnalysis], client: ChatModel, model: str,
    existing_chapters: list[Chapter] | None = None,
    checkpoint: Callable[[list[Chapter]], None] | None = None,
) -> list[Chapter]:
    positions = {row.unit_id: index for index, row in enumerate(units)}
    maps_by_id = {row.unit_id: row for row in maps}
    result = list(existing_chapters or [])
    for chapter in outline[len(result):]:
        left, right = positions[chapter.start_unit], positions[chapter.end_unit]
        source_units = units[left:right + 1]
        chapter_visuals = [asdict(row) for row in visuals
                           if source_units[0].start <= row.timestamp < source_units[-1].end]
        evidence = {
            "outline": asdict(chapter),
            "units": [asdict(row) for row in source_units],
            "content_maps": [asdict(maps_by_id[row.unit_id]) for row in source_units],
            "visuals": chapter_visuals,
        }
        prompt = (
            "Write a detailed Chinese teaching chapter grounded only in this evidence. Reconstruct the reasoning: "
            "motivation/problem, intuition, mechanism, concrete evidence or example, limitations, and takeaway where "
            "applicable. Do not merely replay the transcript chronologically. Preserve all required units and important "
            "details. Refer to selected visuals by timestamp when they support the explanation; if a formula slide is "
            "available, explain its role from speech context without transcribing the formula. Return JSON with title, "
            "summary (the full chapter prose), and key_points.\n" + json.dumps(evidence, ensure_ascii=False)
        )
        payload = parse_json_content(client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
            response_format={"type": "json_object"},
        ))
        result.append(Chapter(
            title=str(payload.get("title") or chapter.title), start=source_units[0].start,
            end=source_units[-1].end, summary=str(payload.get("summary", "")),
            key_points=[str(value) for value in payload.get("key_points", [])],
        ))
        if checkpoint:
            checkpoint(result)
    return result


def review_coverage(
    units: list[EvidenceUnit], maps: list[ContentMap], chapters: list[Chapter],
    client: ChatModel, model: str,
    existing_reviews: list[dict[str, Any]] | None = None,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    maps_by_id = {row.unit_id: row for row in maps}
    reviews = list(existing_reviews or [])
    for chapter in chapters[len(reviews):]:
        chapter_units = [row for row in units if row.end > chapter.start and row.start < chapter.end]
        prompt = (
            "Audit this single drafted chapter against its high-recall content maps. Report only material omissions "
            "and distortions, not wording preferences. Keep every array to at most 12 concise items. Return JSON with "
            "exactly missing_claims, missing_examples, missing_formulas_or_code, distortions, unsupported_claims, "
            "and coherence_issues.\n" + json.dumps({
                "content_maps": [asdict(maps_by_id[row.unit_id]) for row in chapter_units],
                "chapter": asdict(chapter),
            }, ensure_ascii=False)
        )
        review = parse_json_content(client.chat(
            model=model,
            messages=[{"role": "system", "content": "You are an independent evidence coverage reviewer."},
                      {"role": "user", "content": prompt}],
            max_tokens=4096, response_format={"type": "json_object"},
        ))
        reviews.append({"chapter": chapter.title, **review})
        if checkpoint:
            checkpoint(reviews)
    keys = (
        "missing_claims", "missing_examples", "missing_formulas_or_code", "distortions",
        "unsupported_claims", "coherence_issues",
    )
    return {"chapters": reviews, **{
        key: [f"{row['chapter']}: {item}" for row in reviews for item in row.get(key, [])]
        for key in keys
    }}


def synthesize(chapters: list[Chapter], outline: dict[str, Any], client: ChatModel, model: str) -> dict[str, Any]:
    prompt = (
        "Create a Chinese global synthesis from these completed teaching chapters and their outline. Preserve the "
        "reasoning chain and cross-chapter connections; do not compress away important qualifications. Return JSON "
        "with exactly summary, key_concepts, takeaways. Keep key_concepts and takeaways to at most 12 concise "
        "items each.\n"
        + json.dumps({"outline": outline, "chapters": [asdict(row) for row in chapters]}, ensure_ascii=False)
    )
    report = parse_json_content(client.chat(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
        response_format={"type": "json_object"},
    ))
    report.update({"source_language": "zh", "translation_mode": "monolingual",
                   "prompt_version": AUTHORING_PROMPT_VERSION})
    return report
