import json

import pytest

from bbvs.authoring import (
    build_evidence_units, draft_chapters, map_content, plan_outline,
    review_coverage, select_requested_frames, synthesize,
)
from bbvs.models import (
    Chapter, ContentMap, Frame, KnowledgeControl, KnowledgeForm, KnowledgeImportance,
    KnowledgeItem, KnowledgeRole, OutlineChapter, OutlineTopic, Transcript, TranscriptSegment,
)


class FakeChat:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, **kwargs):
        return json.dumps(next(self.responses), ensure_ascii=False)


def transcript() -> Transcript:
    return Transcript("zh", 240, [
        TranscriptSegment(0, 100, "先解释问题"),
        TranscriptSegment(100, 150, "现在看公式"),
        TranscriptSegment(150, 240, "最后给出结论"),
    ], "manual-subtitle", "platform")


def test_evidence_windows_are_not_fixed_chapters_and_preserve_speech() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=120)
    assert [(row.unit_id, row.start, row.end) for row in units] == [
        ("u_0001", 0, 150), ("u_0002", 150, 240),
    ]
    assert "现在看公式" in units[0].speech


def test_content_map_preserves_formula_signal_without_transcription() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    client = FakeChat([{"maps": [{
        "unit_id": "u_0001", "teaching_goal": "解释注意力",
        "knowledge_items": [], "formulas_or_code": ["画面出现复杂度公式"],
        "visual_requests": [], "uncertainties": [],
    }]}])
    maps = map_content(units, client, "m")
    assert maps[0].formulas_or_code == ["画面出现复杂度公式"]


def test_content_map_separates_machine_control_from_chinese_content_label() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    client = FakeChat([{"maps": [{
        "unit_id": "u_0001", "teaching_goal": "解释连续性",
        "knowledge_items": [{
            "id": "k_001", "content": "连续函数的复合仍然连续",
            "control": {
                "form": "proposition", "role": "main", "importance": "essential",
            },
            "content_label": "定理", "relations": [],
        }],
        "formulas_or_code": [], "visual_requests": [], "uncertainties": [],
    }]}])

    maps = map_content(units, client, "m")

    item = maps[0].knowledge_items[0]
    assert item.control.form.value == "proposition"
    assert item.control.role.value == "main"
    assert item.control.importance.value == "essential"
    assert item.content_label == "定理"


def test_content_map_splits_batch_when_model_json_is_truncated() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=120)

    class TruncatingChat:
        def __init__(self):
            self.calls = 0
            self.output_budgets = []

        def chat(self, **kwargs):
            self.calls += 1
            self.output_budgets.append(kwargs["max_tokens"])
            if self.calls == 1:
                return '{"maps":[{"unit_id":"u_0001","knowledge_items":["unfinished'
            unit_id = "u_0001" if self.calls == 2 else "u_0002"
            return json.dumps({"maps": [{"unit_id": unit_id, "knowledge_items": [{
                "id": f"{unit_id}_k_001", "content": unit_id,
                "control": {"form": "proposition", "role": "main", "importance": "essential"},
                "content_label": "结论", "relations": [],
            }]}]})

    client = TruncatingChat()
    maps = map_content(units, client, "m", batch_size=2)

    assert client.calls == 3
    assert client.output_budgets == [8192, 8192, 8192]
    assert [row.unit_id for row in maps] == ["u_0001", "u_0002"]
    assert [[item.content for item in row.knowledge_items] for row in maps] == [["u_0001"], ["u_0002"]]


def test_formula_signal_selects_local_frames_without_global_limit() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", formulas_or_code=["公式"])]
    frames = [Frame(f"{index}.jpg", index * 10, text=["x"] * index) for index in range(1, 9)]
    selected = select_requested_frames(units, maps, frames)
    assert [row.path for row in selected] == ["6.jpg", "7.jpg", "8.jpg"]


def test_outline_must_cover_units_once_in_order() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=120)
    maps = [ContentMap(row.unit_id) for row in units]
    client = FakeChat([{"chapters": [{
        "title": "只有后半", "start_unit": "u_0002", "end_unit": "u_0002",
        "required_items": [], "visual_requests": [],
    }]}])
    with pytest.raises(ValueError, match="cover all evidence units"):
        plan_outline(units, maps, {}, client, "m")


def test_outline_selects_required_knowledge_items_instead_of_all_window_content() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", knowledge_items=[
        KnowledgeItem(
            "k_core", "核心结论",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "核心结论",
        ),
        KnowledgeItem(
            "k_chat", "大家休息一下",
            KnowledgeControl(KnowledgeForm.CONTEXT, KnowledgeRole.CHATTER, KnowledgeImportance.OPTIONAL),
            "课堂闲聊",
        ),
    ])]
    client = FakeChat([{
        "central_question": "问题", "thesis": "结论", "concept_dependencies": [],
        "must_preserve": ["k_core"],
        "chapters": [{
            "title": "主题", "start_unit": "u_0001", "end_unit": "u_0001",
            "teaching_goal": "解释结论", "visual_requests": [], "topics": [{
                "title": "核心主题", "summary": "核心结论", "required_items": ["k_core"],
                "supporting_items": [],
            }],
        }],
    }])

    _, outline = plan_outline(units, maps, {}, client, "m")

    assert outline[0].topics[0].required_items == ["k_core"]


def test_outline_rejects_optional_chatter_as_required_content() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", knowledge_items=[
        KnowledgeItem(
            "k_core", "核心结论",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "核心结论",
        ),
        KnowledgeItem(
            "k_chat", "大家休息一下",
            KnowledgeControl(KnowledgeForm.CONTEXT, KnowledgeRole.CHATTER, KnowledgeImportance.OPTIONAL),
            "课堂闲聊",
        ),
    ])]
    client = FakeChat([{"chapters": [{
        "title": "主题", "start_unit": "u_0001", "end_unit": "u_0001",
        "visual_requests": [], "topics": [{
            "title": "主题", "summary": "核心结论", "required_items": ["k_core"],
            "supporting_items": ["k_chat"],
        }],
    }]}])

    with pytest.raises(ValueError, match="optional or chatter"):
        plan_outline(units, maps, {}, client, "m")


def test_outline_required_items_are_anchors_not_all_essential_content() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", knowledge_items=[
        KnowledgeItem(
            "k_anchor", "主要结论",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "主要结论",
        ),
        KnowledgeItem(
            "k_detail", "不需在大纲中逐项列出的必要细节",
            KnowledgeControl(KnowledgeForm.REASONING, KnowledgeRole.SUPPORTING, KnowledgeImportance.ESSENTIAL),
            "必要推理",
        ),
    ])]
    client = FakeChat([{"chapters": [{
        "title": "主题", "start_unit": "u_0001", "end_unit": "u_0001",
        "visual_requests": [], "topics": [{
            "title": "主要观点", "summary": "主要结论", "required_items": ["k_anchor"],
            "supporting_items": [],
        }],
    }]}])

    _, outline = plan_outline(units, maps, {}, client, "m")

    assert outline[0].topics[0].required_items == ["k_anchor"]


def test_outline_aggregates_candidate_items_into_bounded_report_topics() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", knowledge_items=[
        KnowledgeItem(
            "k_repeat_1", "需求变更会产生技术债",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "结论",
        ),
        KnowledgeItem(
            "k_repeat_2", "每次需求变更都可能累积技术债",
            KnowledgeControl(KnowledgeForm.REASONING, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "原因",
        ),
        KnowledgeItem(
            "k_example", "课程抵扣规则改动导致返工",
            KnowledgeControl(KnowledgeForm.INSTANCE, KnowledgeRole.SUPPORTING, KnowledgeImportance.USEFUL),
            "例子",
        ),
    ])]
    client = FakeChat([{"chapters": [{
        "title": "需求变更", "start_unit": "u_0001", "end_unit": "u_0001",
        "teaching_goal": "解释技术债", "visual_requests": [],
        "topics": [{
            "title": "需求变更与技术债",
            "summary": "需求变更会在现有设计上累积长期成本。",
            "required_items": ["k_repeat_1", "k_repeat_2"],
            "supporting_items": ["k_example"],
        }],
    }]}])

    _, outline = plan_outline(units, maps, {}, client, "m")

    assert outline[0].topics == [OutlineTopic(
        "需求变更与技术债", "需求变更会在现有设计上累积长期成本。",
        ["k_repeat_1", "k_repeat_2"], ["k_example"],
    )]


def test_authoring_long_prose_is_requested_as_plain_text() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)

    class MarkdownChat:
        def __init__(self): self.requests = []
        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return "标题：注意力机制\n\n内容摘要：正文中可以直接使用引号 \" 和代码。"

    client = MarkdownChat()
    chapters = draft_chapters(
        units, [ContentMap("u_0001")],
        [OutlineChapter("注意力", "u_0001", "u_0001")], [], client, "m",
    )
    report = synthesize(chapters, {}, client, "m")

    assert chapters[0].title == "注意力机制"
    assert '引号 "' in chapters[0].summary
    assert report.startswith("标题：注意力机制")
    assert all("Do not use Markdown syntax" in request["messages"][0]["content"] for request in client.requests)
    assert all("response_format" not in request for request in client.requests)


def test_global_synthesis_prioritizes_core_knowledge_without_replaying_chapters() -> None:
    class CapturingChat:
        def __init__(self): self.requests = []
        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return "# 报告\n\n## 核心知识\n\n结论。"

    client = CapturingChat()
    synthesize([Chapter("第一章", 0, 10, "## 第一章\n\n详细内容")], {}, client, "m")

    prompt = client.requests[0]["messages"][0]["content"]
    assert "Do not replay or recap every chapter" in prompt
    assert "omit anecdotes and chatter" in prompt


def test_chapter_drafting_uses_control_enums_to_build_reader_facing_sections() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    content = ContentMap("u_0001", knowledge_items=[
        KnowledgeItem(
            "k_core", "连续函数的复合仍然连续",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "定理",
        ),
        KnowledgeItem(
            "k_example", "用二次函数演示",
            KnowledgeControl(KnowledgeForm.INSTANCE, KnowledgeRole.SUPPORTING, KnowledgeImportance.USEFUL),
            "例子",
        ),
        KnowledgeItem(
            "k_chat", "大家能听清吗",
            KnowledgeControl(KnowledgeForm.CONTEXT, KnowledgeRole.CHATTER, KnowledgeImportance.OPTIONAL),
            "课堂闲聊",
        ),
        KnowledgeItem(
            "k_unselected", "另一个没有被大纲选中的有用细节",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "未选细节",
        ),
    ])

    class CapturingChat:
        def __init__(self): self.requests = []
        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return "## 连续性\n\n### 核心知识\n\n复合仍然连续。"

    client = CapturingChat()
    draft_chapters(
        units, [content], [OutlineChapter(
            "连续性", "u_0001", "u_0001", topics=[OutlineTopic(
                "复合连续性", "连续函数的复合仍然连续。",
                ["k_core"], ["k_example"],
            )],
        )], [], client, "m",
    )

    prompt = client.requests[0]["messages"][0]["content"]
    assert '"required_items": [{"id": "k_core"' in prompt
    assert '"supporting_items": [{"id": "k_example"' in prompt
    assert "大家能听清吗" not in prompt
    assert "另一个没有被大纲选中的有用细节" not in prompt


def test_coverage_review_treats_optional_and_chatter_omission_as_expected() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", knowledge_items=[
        KnowledgeItem(
            "k_core", "核心结论",
            KnowledgeControl(KnowledgeForm.PROPOSITION, KnowledgeRole.MAIN, KnowledgeImportance.ESSENTIAL),
            "核心结论",
        ),
        KnowledgeItem(
            "k_chat", "大家休息一下",
            KnowledgeControl(KnowledgeForm.CONTEXT, KnowledgeRole.CHATTER, KnowledgeImportance.OPTIONAL),
            "课堂闲聊",
        ),
    ])]

    class CapturingChat(FakeChat):
        def __init__(self):
            super().__init__([{
                "missing_essential": [], "lost_reasoning": [], "distortions": [],
                "unsupported_claims": [], "coherence_issues": [],
            }])
            self.requests = []
        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return super().chat(**kwargs)

    client = CapturingChat()
    review = review_coverage(
        units, maps, [OutlineChapter("章节", "u_0001", "u_0001", topics=[OutlineTopic(
            "核心主题", "核心结论", ["k_core"], [],
        )])], [Chapter("章节", 0, 240, "标题：章节\n\n核心结论")], client, "m",
    )

    prompt = client.requests[0]["messages"][1]["content"]
    assert "Unselected candidate items must not be requested" in prompt
    assert "大家休息一下" not in prompt
    assert review["missing_essential"] == []
