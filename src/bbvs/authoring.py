from __future__ import annotations

import json
from dataclasses import asdict
from collections.abc import Callable
from typing import Any

from .errors import BBVSError
from .llm import ChatModel, parse_json_content
from .markdown_artifacts import text_title
from .models import (
    Chapter, ContentMap, EvidenceUnit, Frame, KnowledgeControl, KnowledgeForm,
    KnowledgeImportance, KnowledgeItem, KnowledgeRelation, KnowledgeRelationType,
    KnowledgeRole, OutlineChapter, OutlineTopic, Transcript, VisualAnalysis,
)

AUTHORING_PROMPT_VERSION = "knowledge-topics-text-v4"
MAX_TOPICS_PER_CHAPTER = 4
MAX_REQUIRED_ITEMS_PER_TOPIC = 3
MAX_SUPPORTING_ITEMS_PER_TOPIC = 2


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
        "knowledge drive the outline. Aggregate repeated or closely related candidate items across units into reader-"
        "level topics; a topic is a synthesis, not a restatement of one sentence. Aim for 2-4 topics per chapter and "
        f"never exceed {MAX_TOPICS_PER_CHAPTER}. Each topic may cite at most {MAX_REQUIRED_ITEMS_PER_TOPIC} required "
        f"items as core evidence and at most {MAX_SUPPORTING_ITEMS_PER_TOPIC} supporting items. Supporting items must "
        "attach to exactly one topic and be selected only when they materially improve understanding. Never select "
        "chatter or optional items. Keep concept_dependencies to at most 12 short strings. Return JSON with "
        "exactly central_question, thesis, concept_dependencies, must_preserve, and chapters. Each chapter uses "
        "exactly title, start_unit, end_unit, teaching_goal, topics, visual_requests. Each topic uses exactly title, "
        "summary, required_items, supporting_items. "
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
        topics=[OutlineTopic(
            title=str(topic.get("title", "")), summary=str(topic.get("summary", "")),
            required_items=[str(value) for value in topic.get("required_items", [])],
            supporting_items=[str(value) for value in topic.get("supporting_items", [])],
        ) for topic in row.get("topics", []) if isinstance(topic, dict)],
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
        if len(chapter.topics) > MAX_TOPICS_PER_CHAPTER:
            raise ValueError(f"outline chapter exceeds {MAX_TOPICS_PER_CHAPTER} report topics")
        topic_required = [item_id for topic in chapter.topics for item_id in topic.required_items]
        topic_supporting = [item_id for topic in chapter.topics for item_id in topic.supporting_items]
        for topic in chapter.topics:
            if not topic.title or not topic.summary or not topic.required_items:
                raise ValueError("every report topic needs title, summary, and required items")
            if len(topic.required_items) > MAX_REQUIRED_ITEMS_PER_TOPIC:
                raise ValueError("report topic has too many required items")
            if len(topic.supporting_items) > MAX_SUPPORTING_ITEMS_PER_TOPIC:
                raise ValueError("report topic has too many supporting items")
        for item_id in topic_required + topic_supporting:
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


def _report_topics(maps: list[ContentMap], chapter: OutlineChapter) -> list[dict[str, Any]]:
    item_by_id: dict[str, tuple[KnowledgeItem, str]] = {}
    for content_map in maps:
        for item in content_map.knowledge_items:
            item_by_id[item.id] = (item, content_map.unit_id)

    def selected(item_ids: list[str]) -> list[dict[str, Any]]:
        result = []
        for item_id in item_ids:
            item, unit_id = item_by_id[item_id]
            result.append({**asdict(item), "source_unit": unit_id})
        return result

    return [{
        "title": topic.title,
        "summary": topic.summary,
        "required_items": selected(topic.required_items),
        "supporting_items": selected(topic.supporting_items),
    } for topic in chapter.topics]


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
        report_topics = _report_topics(source_maps, chapter)
        selected_unit_ids = {
            item["source_unit"] for topic in report_topics
            for key in ("required_items", "supporting_items") for item in topic[key]
        }
        selected_units = [row for row in source_units if row.unit_id in selected_unit_ids]
        chapter_visuals = [asdict(row) for row in visuals
                           if source_units[0].start <= row.timestamp < source_units[-1].end]
        map_context = []
        for content_map in source_maps:
            payload = asdict(content_map)
            payload.pop("knowledge_items", None)
            map_context.append(payload)
        evidence = {
            "outline": asdict(chapter),
            "selected_source_units": [asdict(row) for row in selected_units],
            "report_topics": report_topics,
            "content_map_context": map_context,
            "visuals": chapter_visuals,
        }
        prompt = (
            "Write a concise Chinese teaching chapter grounded only in this evidence. Treat report_topics as the "
            "complete content plan. Write one coherent explanation per topic by synthesizing its required_items; do not "
            "describe those items one by one. Use supporting_items only inside their parent topic and only when needed "
            "for comprehension. Content not selected into report_topics must not appear. Never expose English control "
            "enum names. Do not replay the transcript chronologically or repeat equivalent items. Refer to visuals by "
            "timestamp when they support the explanation; if a formula slide is "
            "available, explain its role from speech context without transcribing the formula. Return plain text only. "
            "Do not use Markdown syntax such as #, *, -, backticks, or tables. Start with `标题：...`, then use plain "
            "section labels such as `核心知识：`, `理解辅助：`, and `结论：`; omit empty sections.\n"
            + json.dumps(evidence, ensure_ascii=False)
        )
        text = client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).strip()
        result.append(Chapter(
            title=text_title(text, chapter.title), start=source_units[0].start,
            end=source_units[-1].end, summary=text,
        ))
        if checkpoint:
            checkpoint(result)
    return result


def review_coverage(
    units: list[EvidenceUnit], maps: list[ContentMap], outline: list[OutlineChapter],
    chapters: list[Chapter],
    client: ChatModel, model: str,
    existing_reviews: list[dict[str, Any]] | None = None,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    reviews = list(existing_reviews or [])
    for index, chapter in enumerate(chapters[len(reviews):], len(reviews)):
        planned = outline[index]
        prompt = (
            "Audit this drafted chapter only against its selected report topics. Missing a selected topic or its "
            "required core evidence, a broken reasoning chain, distortions, and unsupported claims are errors. "
            "Unselected candidate items must not be requested or reported as missing. A selected supporting item "
            "matters only when comprehension suffers. Ignore wording preferences. Keep every array to at most 12 "
            "concise items. Return JSON with "
            "exactly missing_essential, lost_reasoning, distortions, unsupported_claims, and coherence_issues.\n"
            + json.dumps({
                "report_topics": _report_topics(maps, planned),
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
        "and omit anecdotes and chatter. Return plain text only, with `标题：`, `内容摘要：`, `核心知识：`, and "
        "`核心结论：` as plain section labels. Do not use Markdown syntax such as #, *, -, backticks, or tables. "
        "Keep each section concise and do not enumerate every source item.\n"
        + json.dumps({"outline": outline, "chapters": [asdict(row) for row in chapters]}, ensure_ascii=False)
    )
    return client.chat(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).strip()
