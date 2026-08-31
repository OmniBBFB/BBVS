import json
from dataclasses import asdict
from pathlib import Path

from bbvs.config import Endpoint, Services
from bbvs.io import read_json, write_json
from bbvs.models import Chapter, Frame, Term, Transcript, TranscriptSegment, VisualAnalysis
from bbvs.pipeline import VideoPipeline
from bbvs.summarize import summarize_timeline
from bbvs.terminology import discover_terms
from bbvs.timeline import build_timeline
from bbvs.verification import suspect_segments, verify_transcript
from bbvs.vision import translate_visual_summaries


class FakeChat:
    def __init__(self, responses): self.responses = iter(responses)
    def chat(self, **kwargs): return json.dumps(next(self.responses))


class RawFakeChat:
    def __init__(self, responses): self.responses = iter(responses)
    def chat(self, **kwargs): return next(self.responses)


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


def test_verifier_recovers_from_truncated_batch_by_splitting_it() -> None:
    transcript = Transcript("zh", 2, [
        TranscriptSegment(0, 1, "汤伟", confidence=0.7),
        TranscriptSegment(1, 2, "卡拉马佐夫", confidence=0.7),
    ], "e", "m")
    client = RawFakeChat([
        '{"corrections":[{"original":"汤伟","evidence":["unfinished',
        json.dumps({"corrections": [{
            "start": 0, "end": 1, "original": "汤伟", "corrected": "陀思妥耶夫斯基",
            "confidence": 0.95, "evidence": ["known term"],
        }]}),
        json.dumps({"corrections": []}),
    ])

    verified, changes = verify_transcript(transcript, [], client, "m", batch_size=2)

    assert verified.segments[0].text == "陀思妥耶夫斯基"
    assert len(changes) == 1


def test_verifier_checkpoints_completed_batches_and_resumes() -> None:
    transcript = Transcript("zh", 3, [
        TranscriptSegment(0, 1, "第一段", confidence=0.7),
        TranscriptSegment(1, 2, "第二段", confidence=0.7),
        TranscriptSegment(2, 3, "汤伟", confidence=0.7),
    ], "e", "m")
    saved = []
    client = FakeChat([{"corrections": [{
        "start": 2, "end": 3, "original": "汤伟", "corrected": "陀思妥耶夫斯基",
        "confidence": 0.95, "evidence": ["known term"],
    }]}])

    verified, changes = verify_transcript(
        transcript, [], client, "m", batch_size=2, processed_suspects=2,
        checkpoint=lambda count, rows: saved.append((count, list(rows))),
    )

    assert verified.segments[2].text == "陀思妥耶夫斯基"
    assert saved == [(3, changes)]


def test_analysis_resumes_transcript_verification_from_disk(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    transcript = Transcript("zh", 3, [
        TranscriptSegment(0, 1, "错一", confidence=0.7),
        TranscriptSegment(1, 2, "正确", confidence=0.7),
        TranscriptSegment(2, 3, "汤伟", confidence=0.7),
    ], "e", "m")
    write_json(run_dir / "source/metadata.json", {})
    write_json(run_dir / "asr-e-m.json", asdict(transcript))
    write_json(run_dir / "ocr-e.json", [])
    write_json(run_dir / "analysis/terminology.json", [])
    write_json(run_dir / "analysis/verification-progress.json", {
        "processed_suspects": 2,
        "corrections": [{
            "start": 0, "end": 1, "original": "错一", "corrected": "正确一",
            "confidence": 0.95, "evidence": ["checkpointed"],
        }],
    })
    pipeline = VideoPipeline(Services(Endpoint("http://unused/v1", "m")))
    pipeline.llm = FakeChat([{"corrections": [{
        "start": 2, "end": 3, "original": "汤伟", "corrected": "陀思妥耶夫斯基",
        "confidence": 0.95, "evidence": ["known term"],
    }]}])

    pipeline.analyze(run_dir, verify=True)

    verified = read_json(run_dir / "analysis/verified-transcript.json")
    assert [row["text"] for row in verified["segments"]] == ["正确一", "正确", "陀思妥耶夫斯基"]


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


def test_final_summary_request_does_not_repeat_chapter_translations() -> None:
    timeline = build_timeline(
        Transcript("en", 60, [TranscriptSegment(0, 60, "Evidence")], "e", "m"), [], []
    )

    class CapturingChat:
        def __init__(self): self.calls = []
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return json.dumps({
                    "title": "Title", "title_zh": "标题", "summary": "Summary",
                    "summary_zh": "摘要", "key_points": ["Point"], "key_points_zh": ["要点"],
                })
            return json.dumps({
                "summary": "Final", "summary_zh": "最终", "key_concepts": [],
                "key_concepts_zh": [], "takeaways": [], "takeaways_zh": [],
            })

    client = CapturingChat()
    summarize_timeline(timeline, client, "m")
    final_call = client.calls[-1]
    final_prompt = final_call["messages"][0]["content"]
    assert '"summary_zh": "摘要"' not in final_prompt
    assert '"key_points_zh"' not in final_prompt
    assert final_call["max_tokens"] == 1536


def test_summary_prompts_request_first_person_without_mechanical_repetition() -> None:
    timeline = build_timeline(
        Transcript("en", 60, [TranscriptSegment(0, 60, "Evidence")], "e", "m"), [], []
    )

    class CapturingChat:
        def __init__(self): self.prompts = []
        def chat(self, **kwargs):
            self.prompts.append(kwargs["messages"][0]["content"])
            if len(self.prompts) == 1:
                return json.dumps({
                    "title": "Topic", "title_zh": "主题", "summary": "I explain it.",
                    "summary_zh": "我解释了它。", "key_points": [], "key_points_zh": [],
                })
            return json.dumps({
                "summary": "I conclude.", "summary_zh": "我的结论。", "key_concepts": [],
                "key_concepts_zh": [], "takeaways": [], "takeaways_zh": [],
            })

    client = CapturingChat()
    summarize_timeline(timeline, client, "m")

    assert all("first-person" in prompt for prompt in client.prompts)
    assert all("mechanically" in prompt for prompt in client.prompts)


def test_chinese_summary_does_not_request_or_return_duplicate_translation_fields() -> None:
    timeline = build_timeline(
        Transcript("zh", 60, [TranscriptSegment(0, 60, "这是中文内容")], "e", "m"), [], []
    )

    class ChineseChat:
        def __init__(self): self.calls = []
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return json.dumps({"title": "标题", "summary": "摘要", "key_points": ["要点"]})
            return json.dumps({"summary": "总摘要", "key_concepts": ["切实的爱"], "takeaways": ["结论"]})

    client = ChineseChat()
    chapters, report = summarize_timeline(timeline, client, "m", source_language="zh")
    assert chapters[0].summary_zh == ""
    assert "summary_zh" not in report
    assert all("_zh" not in call["messages"][0]["content"] for call in client.calls)
