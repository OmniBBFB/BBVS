from bbvs.models import Frame, Term, Transcript, TranscriptSegment
from bbvs.retrieval import search
from bbvs.timeline import build_timeline
from bbvs.vision import select_visual_frames


class FakeEmbedder:
    def embed(self, *, model, texts):
        return [[1.0, 0.0]] + [[1.0, 0.0], [0.0, 1.0]][: len(texts) - 1]


class FakeReranker:
    def rerank(self, *, model, query, documents, top_n=None):
        return [(0, 0.9)]


def transcript() -> Transcript:
    return Transcript("en", 120.0, [
        TranscriptSegment(0, 50, "Money demand depends on interest rates."),
        TranscriptSegment(61, 100, "Look at this chart on the right."),
    ], "fake", "fake")


def test_timeline_aligns_modalities_and_terms() -> None:
    timeline = build_timeline(
        transcript(), [Frame("a.jpg", 10, text=["Money Demand"])],
        [Term("interest rates", "concept")], window_seconds=60,
    )
    assert len(timeline) == 2
    assert timeline[0].screen_text == ["Money Demand"]
    assert timeline[0].terms == ["interest rates"]


def test_visual_router_selects_informative_and_referenced_frames() -> None:
    frames = [Frame("a.jpg", 10, text=[str(i) for i in range(8)]), Frame("b.jpg", 70)]
    assert [row.path for row in select_visual_frames(frames, transcript())] == ["a.jpg", "b.jpg"]


def test_visual_router_spreads_budget_across_full_video() -> None:
    frames = [Frame(f"{index}.jpg", index * 30, text=[str(i) for i in range(8)]) for index in range(10)]
    selected = select_visual_frames(frames, transcript(), max_frames=3)
    assert [row.timestamp for row in selected] == [0, 120, 270]


def test_search_uses_embedding_then_reranker() -> None:
    timeline = build_timeline(transcript(), [], [], window_seconds=60)
    hits = search("money", timeline, FakeEmbedder(), "e", FakeReranker(), "r")
    assert hits[0].segment.start == 0
