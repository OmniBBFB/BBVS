import json
from dataclasses import asdict
from pathlib import Path

from bbvs.config import Endpoint, Services
from bbvs.authoring import AUTHORING_PROMPT_VERSION
from bbvs.io import read_json, write_json
from bbvs.models import Chapter, Frame, Term, Transcript, TranscriptSegment, VisualAnalysis
from bbvs.pipeline import VideoPipeline
from bbvs.summarize import summarize_timeline
from bbvs.terminology import discover_terms
from bbvs.timeline import build_timeline
from bbvs.verification import suspect_segments, verify_transcript
from bbvs.vision import translate_visual_summaries


class FakeChat:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        response = next(self.responses)
        return response if isinstance(response, str) else json.dumps(response)


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


def test_terminology_normalizes_string_evidence_and_disables_thinking() -> None:
    client = FakeChat([{"terms": [{
        "canonical_name": "Git", "category": "concept",
        "confidence": 0.8, "evidence": ["Git repository"],
    }]}])

    terms = discover_terms({"title": "Git"}, [], client, "m")

    assert terms[0].evidence[0].source == "model"
    assert terms[0].evidence[0].text == "Git repository"
    assert client.requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


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


def test_analysis_resumes_nested_knowledge_items_from_content_map_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    transcript_path = run_dir / "asr/transcript.json"
    frames_path = run_dir / "ocr/frames.json"
    write_json(run_dir / "source/metadata.json", {})
    write_json(transcript_path, asdict(Transcript(
        "zh", 10, [TranscriptSegment(0, 10, "连续函数的复合仍然连续")], "e", "m",
    )))
    write_json(frames_path, [])
    write_json(analysis_dir / "terminology.json", [])
    write_json(analysis_dir / "authoring-version.json", {"prompt_version": AUTHORING_PROMPT_VERSION})
    write_json(analysis_dir / "evidence-units.json", [{
        "unit_id": "u_0001", "start": 0, "end": 10,
        "speech": "连续函数的复合仍然连续", "screen_text": [], "source_chapter": None,
    }])
    write_json(analysis_dir / "content-map.json", [{
        "unit_id": "u_0001", "teaching_goal": "说明复合连续性",
        "knowledge_items": [{
            "id": "k_001", "content": "连续函数的复合仍然连续",
            "control": {"form": "proposition", "role": "main", "importance": "essential"},
            "content_label": "定理", "relations": [],
        }],
        "formulas_or_code": [], "visual_requests": [], "uncertainties": [],
    }])
    write_json(analysis_dir / "outline.json", {
        "central_question": "复合是否保持连续性",
        "chapters": [{
            "title": "复合连续性", "start_unit": "u_0001", "end_unit": "u_0001",
            "teaching_goal": "说明定理", "visual_requests": [], "topics": [{
                "title": "复合连续性", "summary": "连续函数的复合仍连续",
                "required_items": ["k_001"], "supporting_items": [],
            }],
        }],
    })
    pipeline = VideoPipeline(Services(Endpoint("http://unused/v1", "m")))
    pipeline.llm = FakeChat([
        "标题：复合连续性\n\n核心知识：连续函数的复合仍然连续。",
        {"missing_essential": [], "distortions": [], "unsupported_claims": [], "coherence_issues": []},
        "标题：连续性报告\n\n核心知识：复合连续。",
    ])

    pipeline.analyze(
        run_dir, summarize=True, authoring=True, transcript_path=transcript_path,
        frames_path=frames_path, analysis_dir=analysis_dir,
    )

    chapter_prompt = pipeline.llm.requests[0]["messages"][0]["content"]
    assert '"content_label": "定理"' in chapter_prompt
    assert (analysis_dir / "report.txt").read_text(encoding="utf-8").startswith("标题：连续性报告")


def test_analysis_rebuilds_authoring_outputs_when_prompt_version_is_stale(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    transcript_path = run_dir / "asr/transcript.json"
    frames_path = run_dir / "ocr/frames.json"
    write_json(run_dir / "source/metadata.json", {})
    write_json(transcript_path, asdict(Transcript(
        "zh", 10, [TranscriptSegment(0, 10, "核心结论")], "e", "m",
    )))
    write_json(frames_path, [])
    write_json(analysis_dir / "terminology.json", [])
    write_json(analysis_dir / "authoring-version.json", {"prompt_version": "old"})
    write_json(analysis_dir / "status.json", {
        "complete": True, "authoring_prompt_version": "old",
    })
    (analysis_dir / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "report.md").write_text("# 旧报告\n", encoding="utf-8")
    pipeline = VideoPipeline(Services(Endpoint("http://unused/v1", "m")))
    pipeline.llm = FakeChat([
        {"maps": [{
            "unit_id": "u_0001", "teaching_goal": "解释结论",
            "knowledge_items": [{
                "id": "u_0001_k_001", "content": "核心结论",
                "control": {"form": "proposition", "role": "main", "importance": "essential"},
                "content_label": "核心结论", "relations": [],
            }],
            "formulas_or_code": [], "visual_requests": [], "uncertainties": [],
        }]},
        {
            "central_question": "什么是核心结论", "thesis": "核心结论", "concept_dependencies": [],
                "must_preserve": ["u_0001_k_001"], "chapters": [{
                    "title": "核心知识", "start_unit": "u_0001", "end_unit": "u_0001",
                    "teaching_goal": "解释结论", "visual_requests": [], "topics": [{
                        "title": "核心结论", "summary": "解释核心结论",
                        "required_items": ["u_0001_k_001"], "supporting_items": [],
                    }],
            }],
        },
        "标题：核心知识\n\n核心知识：新结论。",
        {"missing_essential": [], "lost_reasoning": [], "distortions": [],
         "unsupported_claims": [], "coherence_issues": []},
        "标题：新报告\n\n核心知识：新结论。",
    ])

    pipeline.analyze(
        run_dir, summarize=True, authoring=True, transcript_path=transcript_path,
        frames_path=frames_path, analysis_dir=analysis_dir,
    )

    assert (analysis_dir / "report.txt").read_text(encoding="utf-8").startswith("标题：新报告")
    assert read_json(analysis_dir / "authoring-version.json") == {
        "prompt_version": AUTHORING_PROMPT_VERSION,
    }


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
    client = FakeChat(["## Money\n\nDemand\n\n## 中文\n\n需求", "# Report\n\nLecture summary"])
    chapters, report = summarize_timeline(timeline, client, "m")
    assert chapters[0].title == "Money"
    assert "需求" in chapters[0].summary
    assert "Lecture summary" in report
    assert all("response_format" not in request for request in client.requests)


def test_hierarchical_summary_checkpoints_and_resumes() -> None:
    timeline = build_timeline(
        Transcript("en", 600, [TranscriptSegment(0, 600, "Money demand")], "e", "m"), [], [],
        window_seconds=60,
    )
    existing = Chapter(
        "First", 0, 300, "Original", [], "第一章", "原文", [],
    )
    checkpoints = []
    client = FakeChat(["## Second\n\nMore\n\n## 中文\n\n更多", "# All\n\n全部"])
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
    assert final_call["max_tokens"] == 4096


def test_final_summary_has_enough_output_budget_to_finish_markdown() -> None:
    timeline = build_timeline(
        Transcript("zh", 60, [TranscriptSegment(0, 60, "市场分析")], "e", "m"), [], []
    )

    class BudgetSensitiveChat:
        def __init__(self): self.calls = []
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return "## 市场\n\n震荡"
            if kwargs["max_tokens"] < 3072:
                return "很长的市场总结" * 200
            return "# 总结\n\n我总结市场走势。"

    client = BudgetSensitiveChat()
    _, report = summarize_timeline(timeline, client, "m", source_language="zh")

    assert "我总结市场走势。" in report
    assert "不超过 500 个汉字" in client.calls[-1]["messages"][0]["content"]


def test_summary_prompts_request_first_person_without_mechanical_repetition() -> None:
    timeline = build_timeline(
        Transcript("en", 60, [TranscriptSegment(0, 60, "Evidence")], "e", "m"), [], []
    )

    class CapturingChat:
        def __init__(self): self.prompts = []
        def chat(self, **kwargs):
            self.prompts.append(kwargs["messages"][0]["content"])
            if len(self.prompts) == 1:
                return "## Topic\n\nI explain it.\n\n## 中文\n\n我解释了它。"
            return "# Summary\n\nI conclude.\n\n## 中文\n\n我的结论。"

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
                return "## 标题\n\n摘要\n\n- 要点"
            return "# 总摘要\n\n## 关键概念\n\n- 切实的爱\n\n## 结论\n\n- 结论"

    client = ChineseChat()
    chapters, report = summarize_timeline(timeline, client, "m", source_language="zh")
    assert chapters[0].summary_zh == ""
    assert "总摘要" in report
    assert all("_zh" not in call["messages"][0]["content"] for call in client.calls)
