from __future__ import annotations

import json
from typing import Any

from .llm import ChatModel, parse_json_content
from .models import Evidence, Frame, Term


def discover_terms(
    metadata: dict[str, Any], frames: list[Frame], client: ChatModel, model: str,
    max_ocr_chars: int = 6_000,
) -> list[Term]:
    ocr_rows = []
    for frame in frames:
        if frame.text:
            ocr_rows.append(f"[{frame.timestamp:.1f}s] {' | '.join(frame.text)}")
    evidence = {
        "metadata": {key: metadata.get(key) for key in ("title", "description", "tags")},
        "ocr": "\n".join(ocr_rows)[:max_ocr_chars],
    }
    prompt = f"""Extract at most 12 high-value named entities and domain terms from this video evidence.
Return JSON: {{"terms":[{{"canonical_name":"...","category":"person|organization|course|concept|symbol|other","aliases":[],"confidence":0.0,"evidence":[{{"source":"metadata|ocr","text":"short exact evidence","timestamp":null}}]}}]}}.
Do not invent terms. Deduplicate aliases. Include exactly one evidence item per term and keep its text under 80 characters. Evidence:\n{json.dumps(evidence, ensure_ascii=False)}"""
    payload = parse_json_content(client.chat(
        model=model,
        messages=[{"role": "system", "content": "You extract auditable terminology."}, {"role": "user", "content": prompt}],
        max_tokens=3072,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"},
    ))
    terms = []
    for item in payload.get("terms", []):
        term_evidence = [Evidence(**row) for row in item.get("evidence", [])]
        terms.append(Term(
            canonical_name=str(item["canonical_name"]).strip(),
            category=str(item.get("category", "other")),
            aliases=[str(value) for value in item.get("aliases", [])],
            confidence=float(item.get("confidence", 0.0)),
            evidence=term_evidence,
        ))
    return [term for term in terms if term.canonical_name]


def context_text(terms: list[Term], limit: int = 80) -> str:
    ranked = sorted(terms, key=lambda term: term.confidence, reverse=True)[:limit]
    return ", ".join(term.canonical_name for term in ranked)
