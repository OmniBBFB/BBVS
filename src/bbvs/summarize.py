from __future__ import annotations

import json
from collections.abc import Callable

from .llm import ChatModel, parse_json_content
from .models import Chapter, TimelineSegment


def summarize_timeline(
    timeline: list[TimelineSegment], client: ChatModel, model: str,
    segments_per_chapter: int = 5,
    existing_chapters: list[Chapter] | None = None,
    checkpoint: Callable[[list[Chapter]], None] | None = None,
) -> tuple[list[Chapter], dict]:
    chapters = list(existing_chapters or [])
    start_offset = len(chapters) * segments_per_chapter
    for offset in range(start_offset, len(timeline), segments_per_chapter):
        group = timeline[offset:offset + segments_per_chapter]
        evidence = [{
            "start": row.start, "end": row.end, "speech": row.speech,
            "screen_text": row.screen_text[:30],
            "visual": [item.summary for item in row.visuals],
        } for row in group]
        prompt = (
            "Summarize this chronological evidence without inventing facts. Preserve the original language in "
            "title, summary, and key_points, then translate each field into Chinese. Return JSON with exactly "
            "title, title_zh, summary, summary_zh, key_points, key_points_zh. The two key-point arrays must be "
            "aligned item by item.\n" + json.dumps(evidence, ensure_ascii=False)
        )
        payload = parse_json_content(client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=2048,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ))
        chapters.append(Chapter(
            title=str(payload.get("title", f"Chapter {len(chapters) + 1}")),
            start=group[0].start, end=group[-1].end,
            summary=str(payload.get("summary", "")),
            key_points=[str(value) for value in payload.get("key_points", [])],
            title_zh=str(payload.get("title_zh", "")),
            summary_zh=str(payload.get("summary_zh", "")),
            key_points_zh=[str(value) for value in payload.get("key_points_zh", [])],
        ))
        if checkpoint:
            checkpoint(chapters)
    condensed = [{
        "title": row.title, "title_zh": row.title_zh, "start": row.start, "end": row.end,
        "summary": row.summary, "summary_zh": row.summary_zh,
        "key_points": row.key_points, "key_points_zh": row.key_points_zh,
    } for row in chapters]
    final_prompt = (
        "Create an evidence-grounded video report. Preserve the original language and provide an aligned Chinese "
        "translation. Return JSON with exactly summary, summary_zh, key_concepts, key_concepts_zh, takeaways, "
        "takeaways_zh, chapters. Original and _zh arrays must be aligned item by item.\n"
        + json.dumps(condensed, ensure_ascii=False)
    )
    report = parse_json_content(client.chat(
        model=model, messages=[{"role": "user", "content": final_prompt}], max_tokens=2048,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ))
    return chapters, report
