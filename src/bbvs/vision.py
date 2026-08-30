from __future__ import annotations

import base64
import json
from pathlib import Path

from .llm import ChatModel, parse_json_content
from .models import Frame, Transcript, VisualAnalysis

_VISUAL_REFERENCES = ("here", "there", "this chart", "this graph", "on the right", "on the left", "这里", "图中", "右边", "左边")


def select_visual_frames(
    frames: list[Frame], transcript: Transcript, max_frames: int = 24, min_gap: float = 20.0,
) -> list[Frame]:
    candidates: list[Frame] = []
    for frame in frames:
        nearby = " ".join(row.text for row in transcript.segments if abs(row.start - frame.timestamp) < 12)
        informative_slide = len(frame.text) >= 8
        referenced = any(value in nearby.casefold() for value in _VISUAL_REFERENCES)
        if (informative_slide or referenced) and (
            not candidates or frame.timestamp - candidates[-1].timestamp >= min_gap
        ):
            candidates.append(frame)
    if len(candidates) <= max_frames:
        return candidates
    if max_frames == 1:
        return [candidates[len(candidates) // 2]]
    indices = [round(index * (len(candidates) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [candidates[index] for index in indices]


def analyze_frames(frames: list[Frame], client: ChatModel, model: str) -> list[VisualAnalysis]:
    analyses = []
    for frame in frames:
        mime = "image/png" if Path(frame.path).suffix.lower() == ".png" else "image/jpeg"
        image = base64.b64encode(Path(frame.path).read_bytes()).decode("ascii")
        prompt = (
            "Return only compact JSON with summary, summary_zh, visual_type, entities, relations. "
            "Preserve the original language in summary and translate it into Chinese in summary_zh. "
            "summary must be at most 60 words; entities and relations must each contain at most 6 short strings. "
            "Capture spatial/chart relationships not already obvious from OCR."
        )
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
            {"type": "text", "text": prompt + "\nOCR: " + " | ".join(frame.text)},
        ]
        payload = parse_json_content(client.chat(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=768, response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ))
        analyses.append(VisualAnalysis(
            timestamp=frame.timestamp, summary=str(payload.get("summary", "")),
            visual_type=str(payload.get("visual_type", "other")),
            summary_zh=str(payload.get("summary_zh", "")),
            entities=[str(value) for value in payload.get("entities", [])],
            relations=[str(value) for value in payload.get("relations", [])],
        ))
    return analyses


def translate_visual_summaries(
    analyses: list[VisualAnalysis], client: ChatModel, model: str, batch_size: int = 8,
) -> list[VisualAnalysis]:
    for start in range(0, len(analyses), batch_size):
        batch = [row for row in analyses[start:start + batch_size] if not row.summary_zh]
        if not batch:
            continue
        rows = [{"timestamp": row.timestamp, "summary": row.summary} for row in batch]
        payload = parse_json_content(client.chat(
            model=model,
            messages=[{"role": "user", "content": (
                "Translate each visual summary faithfully into concise Chinese. Return JSON with translations, "
                "an array containing timestamp and summary_zh in the same order.\n"
                + json.dumps(rows, ensure_ascii=False)
            )}],
            max_tokens=1024, response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ))
        translations = payload.get("translations", [])
        by_timestamp = {round(float(item["timestamp"]), 3): str(item.get("summary_zh", "")) for item in translations}
        for row in batch:
            row.summary_zh = by_timestamp.get(round(row.timestamp, 3), "")
    return analyses
