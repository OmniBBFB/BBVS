from __future__ import annotations

import json
from dataclasses import asdict
from collections.abc import Callable
from typing import Any

from .errors import BBVSError
from .llm import ChatModel, parse_json_content
from .markdown_artifacts import markdown_title
from .models import (
    Chapter, ContentMap, EvidenceUnit, Frame, KnowledgeControl, KnowledgeForm,
    KnowledgeImportance, KnowledgeItem, KnowledgeRelation, KnowledgeRelationType,
    KnowledgeRole, OutlineChapter, Transcript, VisualAnalysis,
)

AUTHORING_PROMPT_VERSION = "knowledge-items-markdown-v3"


def content_map_from_dict(row: dict[str, Any]) -> ContentMap:
    knowledge_items = []
    for index, value in enumerate(row.get("knowledge_items", []), 1):
        control = value.get("control", {})
        relations = [KnowledgeRelation(
            type=KnowledgeRelationType(relation["type"]),
            target=str(relation.get("target", "")),
        ) for relation in value.get("relations", []) if isinstance(relation, dict)]
        knowledge_items.append(KnowledgeItem(
            id=str(value.get("id") or f"{row.get('unit_id', 'unit')}_k_{index:03d}"),
            content=str(value.get("content", "")),
            control=KnowledgeControl(
                form=KnowledgeForm(control["form"]),
                role=KnowledgeRole(control["role"]),
                importance=KnowledgeImportance(control["importance"]),
            ),
            content_label=str(value.get("content_label", "")),
            relations=relations,
        ))
    return ContentMap(
        unit_id=str(row.get("unit_id", "")),
        teaching_goal=str(row.get("teaching_goal", "")),
        knowledge_items=knowledge_items,
        formulas_or_code=[str(value) for value in row.get("formulas_or_code", [])],
        visual_requests=[value for value in row.get("visual_requests", []) if isinstance(value, dict)],
        uncertainties=[str(value) for value in row.get("uncertainties", [])],
    )


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

    def process_batch(batch: list[EvidenceUnit]) -> None:
        prompt = (
            "Create a high-recall content map for every evidence unit. Do not summarize for brevity and do not "
            "invent facts. Extract atomic knowledge_items. Each item uses exactly id, content, control, "
            "content_label, relations. control uses only the English enums form"
            "(definition|proposition|reasoning|procedure|instance|context), role"
            "(main|supporting|anecdote|chatter), and importance(essential|useful|optional). "
            "content_label is a short Chinese reader-facing label chosen freely from the subject matter; it never "
            "controls behavior. Each relation uses type(defines|depends_on|supports|derives_from|explains|"
            "instantiates|refutes|qualifies|applies_to) and target, referring to another item id. Preserve formulas "
            "or code signals and uncertainty. Identify visual requests when seeing the original slide would "
            "materially help; formulas only need their original slide, not transcription. Item ids must be unique "
            "and prefixed with their unit_id. Return JSON with maps, one map per unit, using exactly: unit_id, "
            "teaching_goal, knowledge_items, formulas_or_code, "
            "visual_requests, uncertainties. Each visual request has description, "
            "start, end, importance(required|useful).\n" + json.dumps([asdict(row) for row in batch], ensure_ascii=False)
        )
        response = client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
            response_format={"type": "json_object"},
        )
        try:
            payload = parse_json_content(response)
            by_id = {str(row.get("unit_id")): row for row in payload.get("maps", [])}
            parsed = [content_map_from_dict({**by_id.get(unit.unit_id, {}), "unit_id": unit.unit_id})
                      for unit in batch]
        except (BBVSError, KeyError, TypeError, ValueError) as exc:
            if len(batch) == 1:
                if isinstance(exc, BBVSError):
                    raise
                raise BBVSError(f"invalid content map schema: {exc}") from exc
            midpoint = len(batch) // 2
            process_batch(batch[:midpoint])
            process_batch(batch[midpoint:])
            return
        maps.extend(parsed)
        if checkpoint:
            checkpoint(maps)

    for start in range(0, len(pending), batch_size):
        process_batch(pending[start:start + batch_size])
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
        "fixed time windows, while keeping every chapter backed by a contiguous source range. Let essential main "
        "knowledge drive the outline; include useful supporting items only when they improve comprehension. Never "
        "require chatter or optional items, and merge repeated knowledge. Keep the output bounded: concept_dependencies "
        "must be an array of at most 12 short strings; must_preserve must contain at most 20 essential knowledge item "
        "ids. Return JSON with "
        "exactly central_question, thesis, concept_dependencies, must_preserve, and chapters. Each chapter uses "
        "exactly title, start_unit, end_unit, teaching_goal, required_items, visual_requests. "
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
        required_items=[str(value) for value in row.get("required_items", [])],
        visual_requests=[value for value in row.get("visual_requests", []) if isinstance(value, dict)],
    ) for row in payload.get("chapters", [])]
    _validate_outline(units, maps, chapters)
    return payload, chapters


def _validate_outline(
    units: list[EvidenceUnit], maps: list[ContentMap], chapters: list[OutlineChapter],
) -> None:
    ids = [row.unit_id for row in units]
    positions = {value: index for index, value in enumerate(ids)}
    item_by_id: dict[str, tuple[KnowledgeItem, str]] = {}
    for content_map in maps:
        for item in content_map.knowledge_items:
            if item.id in item_by_id:
                raise ValueError(f"duplicate knowledge item id: {item.id}")
            item_by_id[item.id] = (item, content_map.unit_id)
    covered: list[str] = []
    required: list[str] = []
    for chapter in chapters:
        if chapter.start_unit not in positions or chapter.end_unit not in positions:
            raise ValueError("outline references an unknown evidence unit")
        left, right = positions[chapter.start_unit], positions[chapter.end_unit]
        if left > right:
            raise ValueError("outline chapter range is reversed")
        covered.extend(ids[left:right + 1])
        for item_id in chapter.required_items:
            if item_id not in item_by_id:
                raise ValueError(f"outline references an unknown required item: {item_id}")
            item, source_unit = item_by_id[item_id]
            if not left <= positions[source_unit] <= right:
                raise ValueError(f"required item is outside its chapter range: {item_id}")
            if item.control.role is KnowledgeRole.CHATTER or item.control.importance is KnowledgeImportance.OPTIONAL:
                raise ValueError(f"optional or chatter item cannot be required: {item_id}")
            required.append(item_id)
    if covered != ids:
        raise ValueError("outline chapters must cover all evidence units exactly once in source order")
    if len(required) != len(set(required)):
        raise ValueError("outline required items must not repeat")
    essential = {
        item.id for item, _ in item_by_id.values()
        if item.control.importance is KnowledgeImportance.ESSENTIAL
        and item.control.role is not KnowledgeRole.CHATTER
    }
    missing = essential - set(required)
    if missing:
        raise ValueError(f"outline omits essential required items: {', '.join(sorted(missing))}")


def _report_sections(maps: list[ContentMap]) -> dict[str, list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = {
        "core_knowledge": [],
        "explanations_and_examples": [],
        "supplementary": [],
    }
    for content_map in maps:
        for item in content_map.knowledge_items:
            control = item.control
            if control.role is KnowledgeRole.CHATTER or control.importance is KnowledgeImportance.OPTIONAL:
                continue
            payload = asdict(item)
            payload["source_unit"] = content_map.unit_id
            if control.role is KnowledgeRole.MAIN and control.importance is KnowledgeImportance.ESSENTIAL:
                sections["core_knowledge"].append(payload)
            elif control.role is KnowledgeRole.ANECDOTE or control.form is KnowledgeForm.CONTEXT:
                sections["supplementary"].append(payload)
            else:
                sections["explanations_and_examples"].append(payload)
    return sections


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
        source_maps = [maps_by_id[row.unit_id] for row in source_units]
        chapter_visuals = [asdict(row) for row in visuals
                           if source_units[0].start <= row.timestamp < source_units[-1].end]
        map_context = []
        for content_map in source_maps:
            payload = asdict(content_map)
            payload.pop("knowledge_items", None)
            map_context.append(payload)
        evidence = {
            "outline": asdict(chapter),
            "units": [asdict(row) for row in source_units],
            "report_sections": _report_sections(source_maps),
            "content_map_context": map_context,
            "visuals": chapter_visuals,
        }
        prompt = (
            "Write a concise Chinese teaching chapter grounded only in this evidence. Treat report_sections as the "
            "content plan: explain every core_knowledge item, use explanations_and_examples only when they materially "
            "help understanding, and keep supplementary brief. Content omitted from report_sections, including "
            "optional material and chatter, must not appear. Organize the reader-facing chapter under `### 核心知识`, "
            "`### 解释与例子`, and `### 补充内容` as applicable; omit empty sections. Use each item's "
            "Chinese content_label naturally as a small label when useful, but never expose English control enum names. "
            "Do not replay the transcript chronologically or repeat equivalent items. Refer to selected visuals by "
            "timestamp when they support the explanation; if a formula slide is "
            "available, explain its role from speech context without transcribing the formula. Return only a complete "
            "Markdown chapter. Start with one level-2 heading, use prose and lists naturally, and do not wrap the "
            "document in a code fence.\n" + json.dumps(evidence, ensure_ascii=False)
        )
        markdown = client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).strip()
        result.append(Chapter(
            title=markdown_title(markdown, chapter.title), start=source_units[0].start,
            end=source_units[-1].end, summary=markdown,
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
            "Audit this single drafted chapter against its classified knowledge items. Missing essential items, a "
            "broken reasoning chain, distortions, and unsupported claims are errors. Missing useful supporting items "
            "matters only when comprehension suffers; omission of optional items or chatter is expected and must not "
            "be reported. Ignore wording preferences. Keep every array to at most 12 concise items. Return JSON with "
            "exactly missing_essential, lost_reasoning, distortions, unsupported_claims, and coherence_issues.\n"
            + json.dumps({
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
        "missing_essential", "lost_reasoning", "distortions", "unsupported_claims", "coherence_issues",
    )
    return {"chapters": reviews, **{
        key: [f"{row['chapter']}: {item}" for row in reviews for item in row.get(key, [])]
        for key in keys
    }}


def synthesize(chapters: list[Chapter], outline: dict[str, Any], client: ChatModel, model: str) -> str:
    prompt = (
        "Create a concise Chinese global synthesis from these completed teaching chapters and their outline. Lead "
        "with the central question, core knowledge, reasoning chain, and cross-chapter connections. Do not replay or "
        "recap every chapter. Keep only examples needed to understand a core idea, preserve material qualifications, "
        "and omit anecdotes and chatter. Return only a complete Markdown report with a title, a brief summary, a "
        "core-knowledge section, and a takeaways section. Keep lists to at most 12 concise items each. Do not use a "
        "wrapping code fence.\n"
        + json.dumps({"outline": outline, "chapters": [asdict(row) for row in chapters]}, ensure_ascii=False)
    )
    return client.chat(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).strip()
