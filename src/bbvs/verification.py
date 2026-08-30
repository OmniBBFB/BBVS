from __future__ import annotations

import json
from dataclasses import replace
from typing import Callable

from .errors import BBVSError
from .llm import ChatModel, parse_json_content
from .models import Correction, Term, Transcript, TranscriptSegment
from .terminology import context_text


def _batches(segments: list[TranscriptSegment], size: int):
    for start in range(0, len(segments), size):
        yield segments[start:start + size]


def suspect_segments(
    segments: list[TranscriptSegment], confidence_threshold: float = 0.75,
    word_probability_threshold: float = 0.15,
) -> list[TranscriptSegment]:
    result = []
    for segment in segments:
        low_segment = segment.confidence is not None and segment.confidence < confidence_threshold
        low_word = any(
            word.get("probability") is not None
            and float(word["probability"]) < word_probability_threshold
            for word in segment.words
        )
        if low_segment or low_word:
            result.append(segment)
    return result


def verify_transcript(
    transcript: Transcript, terms: list[Term], client: ChatModel, model: str,
    batch_size: int = 4,
    processed_suspects: int = 0,
    existing_corrections: list[Correction] | None = None,
    checkpoint: Callable[[int, list[Correction]], None] | None = None,
) -> tuple[Transcript, list[Correction]]:
    corrections = list(existing_corrections or [])
    suspects = suspect_segments(transcript.segments)
    processed_suspects = min(max(processed_suspects, 0), len(suspects))

    def verify_batch(batch: list[TranscriptSegment]) -> list[Correction]:
        rows = [{"start": row.start, "end": row.end, "text": row.text} for row in batch]
        prompt = f"""These segments were selected for low ASR confidence. Conservatively find actual errors.
Allowed: names, terminology, homophones, spelling and punctuation. Forbidden: paraphrasing, adding or deleting claims.
Known terminology: {context_text(terms)}
Return JSON {{"corrections":[{{"start":0,"end":0,"original":"exact substring","corrected":"replacement","confidence":0.0,"evidence":["term or reason"]}}]}}.
Only return high-confidence corrections. Keep each evidence reason under 12 words. Segments:\n{json.dumps(rows, ensure_ascii=False)}"""
        content = client.chat(
            model=model,
            messages=[{"role": "system", "content": "You are a conservative ASR verifier."}, {"role": "user", "content": prompt}],
            max_tokens=2048,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        try:
            payload = parse_json_content(content)
        except BBVSError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            return verify_batch(batch[:middle]) + verify_batch(batch[middle:])
        return [Correction(
            start=float(item["start"]), end=float(item["end"]),
            original=str(item["original"]), corrected=str(item["corrected"]),
            confidence=float(item.get("confidence", 0.0)),
            evidence=[str(value) for value in item.get("evidence", [])],
        ) for item in payload.get("corrections", [])]

    completed = processed_suspects
    for batch in _batches(suspects[processed_suspects:], batch_size):
        corrections.extend(verify_batch(batch))
        completed += len(batch)
        if checkpoint:
            checkpoint(completed, corrections)

    updated = [replace(segment) for segment in transcript.segments]
    for correction in corrections:
        if correction.confidence < 0.8 or not correction.original:
            continue
        for segment in updated:
            if segment.start <= correction.start + 0.25 and segment.end >= correction.end - 0.25:
                if correction.original in segment.text:
                    segment.text = segment.text.replace(correction.original, correction.corrected, 1)
                    break
    return replace(transcript, segments=updated), corrections
