from __future__ import annotations

import json
from collections.abc import Callable

from .llm import ChatModel, parse_json_content
from .models import Chapter, TimelineSegment

SUMMARY_PROMPT_VERSION = "first-person-v1"


def summarize_timeline(
    timeline: list[TimelineSegment], client: ChatModel, model: str,
    segments_per_chapter: int = 5,
    existing_chapters: list[Chapter] | None = None,
    checkpoint: Callable[[list[Chapter]], None] | None = None,
    source_language: str | None = None,
) -> tuple[list[Chapter], dict]:
    chinese_source = (source_language or "").casefold().startswith(("zh", "cmn", "yue"))
    chapters = list(existing_chapters or [])
    start_offset = len(chapters) * segments_per_chapter
    for offset in range(start_offset, len(timeline), segments_per_chapter):
        group = timeline[offset:offset + segments_per_chapter]
        evidence = [{
            "start": row.start, "end": row.end, "speech": row.speech,
            "screen_text": row.screen_text[:30],
            "visual": [item.summary for item in row.visuals],
        } for row in group]
        if chinese_source:
            prompt = (
                "以讲述者本人的第一人称视角，用自然中文总结以下时间证据，不得编造事实。"
                "summary 应像我在亲自归纳自己的讲述；不要使用“讲述者”“作者”“本视频”等第三人称或旁观者表述，"
                "也不要为了强调视角而在每句话机械重复“我”。title 使用简洁的主题短语。"
                "返回 JSON，且只包含 title、summary、key_points。"
                "关键概念和必要外文专名应以中文为主，可在括号中保留原文；不要翻译成英文。\n"
                + json.dumps(evidence, ensure_ascii=False)
            )
        else:
            prompt = (
                "Summarize this chronological evidence without inventing facts. Preserve the original language in "
                "title, summary, and key_points, then translate each field into Chinese. Write the summary in the "
                "speaker's first-person voice, as if I am concisely recapping my own explanation. Do not refer to "
                "the speaker, author, presenter, or video in the third person, and do not mechanically begin every "
                "sentence with 'I'. Keep the title as a concise topic phrase. Apply the same perspective to the "
                "Chinese translation. Return JSON with exactly "
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
    # The final synthesis only needs semantic evidence once. Re-sending Chinese
    # translations and timestamps nearly doubles context without adding facts.
    condensed = [{
        "title": row.title, "summary": row.summary, "key_points": row.key_points,
    } for row in chapters]
    if chinese_source:
        final_prompt = (
            "基于以下章节，以讲述者本人的第一人称视角生成简洁的中文内容总结。"
            "summary 应像我在回顾并归纳自己的完整讲述；不要写成“讲述者介绍了”“作者认为”或“本视频讨论了”，"
            "也不要在每句话机械重复“我”。返回 JSON，且只包含 summary、key_concepts、takeaways。"
            "所有概念使用自然中文；必要外文专名可在括号中保留原文，不要另造英文版本。\n"
            + json.dumps(condensed, ensure_ascii=False)
        )
    else:
        final_prompt = (
            "Create an evidence-grounded video report. Preserve the original language and provide an aligned Chinese "
            "translation. Write both summaries in the speaker's first-person voice, as if I am recapping my own "
            "explanation. Never describe the speaker, author, presenter, or video from a third-person observer's "
            "perspective, and avoid mechanically starting every sentence with 'I'. Return JSON with exactly "
            "summary, summary_zh, key_concepts, key_concepts_zh, takeaways, "
            "takeaways_zh. Original and _zh arrays must be aligned item by item. Keep the summary concise.\n"
            + json.dumps(condensed, ensure_ascii=False)
        )
    report = parse_json_content(client.chat(
        model=model, messages=[{"role": "user", "content": final_prompt}], max_tokens=1536,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ))
    report["source_language"] = source_language
    report["translation_mode"] = "monolingual" if chinese_source else "bilingual_zh"
    return chapters, report
