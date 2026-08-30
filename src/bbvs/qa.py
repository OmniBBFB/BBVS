from __future__ import annotations

import json

from .llm import ChatModel, parse_json_content
from .retrieval import SearchHit


def answer_question(query: str, hits: list[SearchHit], client: ChatModel, model: str) -> dict:
    evidence = [{
        "start": hit.segment.start, "end": hit.segment.end,
        "speech": hit.segment.speech, "screen_text": hit.segment.screen_text,
        "visual": [item.summary for item in hit.segment.visuals],
    } for hit in hits]
    prompt = f"""Answer the question using only the supplied video evidence.
Return JSON {{"answer":"...","citations":[{{"start":0,"end":0,"reason":"..."}}],"insufficient_evidence":false}}.
Every factual claim must be supported by a timestamp citation. If evidence is insufficient, say so.
Question: {query}\nEvidence: {json.dumps(evidence, ensure_ascii=False)}"""
    return parse_json_content(client.chat(
        model=model,
        messages=[{"role": "system", "content": "You answer questions with auditable video citations."}, {"role": "user", "content": prompt}],
        max_tokens=2048, response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"},
    ))
