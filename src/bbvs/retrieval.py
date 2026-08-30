from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .models import TimelineSegment


class Embedder(Protocol):
    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, *, model: str, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]: ...


@dataclass(slots=True)
class SearchHit:
    index: int
    score: float
    segment: TimelineSegment


def segment_text(segment: TimelineSegment) -> str:
    visual = " ".join(item.summary for item in segment.visuals)
    return f"[{segment.start:.1f}-{segment.end:.1f}] {segment.speech}\nOCR: {' | '.join(segment.screen_text)}\nVisual: {visual}"


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norm if norm else 0.0


def search(
    query: str, timeline: list[TimelineSegment], embedder: Embedder, embedding_model: str,
    reranker: Reranker | None = None, reranker_model: str | None = None,
    candidate_count: int = 20, top_k: int = 5,
) -> list[SearchHit]:
    documents = [segment_text(row) for row in timeline]
    instructed_query = (
        "Instruct: Retrieve video segments that directly answer the user's question.\n"
        f"Query: {query}"
    )
    vectors = embedder.embed(model=embedding_model, texts=[instructed_query, *documents])
    ranked = sorted(range(len(documents)), key=lambda index: _cosine(vectors[0], vectors[index + 1]), reverse=True)[:candidate_count]
    if reranker and reranker_model:
        order = reranker.rerank(
            model=reranker_model, query=instructed_query,
            documents=[documents[index] for index in ranked], top_n=top_k,
        )
        return [SearchHit(ranked[local], score, timeline[ranked[local]]) for local, score in order[:top_k]]
    return [SearchHit(index, _cosine(vectors[0], vectors[index + 1]), timeline[index]) for index in ranked[:top_k]]
