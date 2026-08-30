import json

from bbvs.models import Chapter, Frame, Term, Transcript, TranscriptSegment, VisualAnalysis
from bbvs.summarize import summarize_timeline
from bbvs.terminology import discover_terms
from bbvs.timeline import build_timeline
from bbvs.verification import suspect_segments, verify_transcript
from bbvs.vision import translate_visual_summaries


class FakeChat:
    def __init__(self, responses): self.responses = iter(responses)
    def chat(self, **kwargs): return json.dumps(next(self.responses))


def test_terminology_is_converted_to_domain_models() -> None:
    client = FakeChat([{"terms": [{
        "canonical_name": "Federal Reserve", "category": "organization",
        "confidence": 0.9, "aliases": ["Fed"],
        "evidence": [{"source": "ocr", "text": "Federal Reserve", "timestamp": 10.0}],
    }]}])
    terms = discover_terms({"title": "Lecture"}, [Frame("x", 10, text=["Federal Reserve"])], client, "m")
    assert terms[0].canonical_name == "Federal Reserve"
    assert terms[0].evidence[0].timestamp == 10.0


def test_verifier_only_applies_high_confidence_exact_replacements() -> None:
    transcript = Transcript("en", 10, [TranscriptSegment(0, 10, "the federal reserved", confidence=0.7)], "e", "m")
    client = FakeChat([{"corrections": [{
        "start": 0, "end": 10, "original": "federal reserved",
        "corrected": "Federal Reserve", "confidence": 0.95, "evidence": ["known term"],
    }]}])
    verified, changes = verify_transcript(transcript, [Term("Federal Reserve", "organization")], client, "m")
    assert verified.segments[0].text == "the Federal Reserve"
    assert len(changes) == 1


def test_verifier_only_selects_low_confidence_segments() -> None:
    rows = [
        TranscriptSegment(0, 1, "good", confidence=0.9, words=[{"probability": 0.9}]),
        TranscriptSegment(1, 2, "bad", confidence=0.7, words=[]),
        TranscriptSegment(2, 3, "word", confidence=0.9, words=[{"probability": 0.1}]),
    ]
    assert [row.text for row in suspect_segments(rows)] == ["bad", "word"]


def test_hierarchical_summary_returns_chapters_and_report() -> None:
    timeline = build_timeline(
        Transcript("en", 60, [TranscriptSegment(0, 60, "Money demand")], "e", "m"), [], []
    )
    client = FakeChat([
        {"title": "Money", "title_zh": "货币", "summary": "Demand", "summary_zh": "需求",
         "key_points": ["rates"], "key_points_zh": ["利率"]},
        {"summary": "Lecture summary", "summary_zh": "课程总结", "key_concepts": ["money"],
         "key_concepts_zh": ["货币"], "takeaways": [], "takeaways_zh": [], "chapters": []},
    ])
    chapters, report = summarize_timeline(timeline, client, "m")
    assert chapters[0].title == "Money"
    assert chapters[0].summary_zh == "需求"
    assert report["summary"] == "Lecture summary"


def test_hierarchical_summary_checkpoints_and_resumes() -> None:
    timeline = build_timeline(
        Transcript("en", 600, [TranscriptSegment(0, 600, "Money demand")], "e", "m"), [], [],
        window_seconds=60,
    )
    existing = Chapter(
        "First", 0, 300, "Original", [], "第一章", "原文", [],
    )
    checkpoints = []
    client = FakeChat([
        {"title": "Second", "title_zh": "第二章", "summary": "More", "summary_zh": "更多",
         "key_points": [], "key_points_zh": []},
        {"summary": "All", "summary_zh": "全部", "key_concepts": [], "key_concepts_zh": [],
         "takeaways": [], "takeaways_zh": [], "chapters": []},
    ])
    chapters, _ = summarize_timeline(
        timeline, client, "m", existing_chapters=[existing],
        checkpoint=lambda rows: checkpoints.append(list(rows)),
    )
    assert [row.title for row in chapters] == ["First", "Second"]
    assert len(checkpoints) == 1 and len(checkpoints[0]) == 2


def test_visual_summary_translation_preserves_original() -> None:
    rows = [VisualAnalysis(10, "A demand curve.", "chart")]
    client = FakeChat([{"translations": [{"timestamp": 10, "summary_zh": "一条需求曲线。"}]}])
    translated = translate_visual_summaries(rows, client, "m")
    assert translated[0].summary == "A demand curve."
    assert translated[0].summary_zh == "一条需求曲线。"
