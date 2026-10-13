from copy import deepcopy
import json

import pytest

from bbvs.document_review import review
from bbvs.models import Chapter, EvidenceUnit
from bbvs.teaching_document import compose


class Chat:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return json.dumps(result) if isinstance(result, dict) else result


def original():
    return {"evidence_ids": ["k"], "figures": [], "chapters": [{"title": "课程", "start": 0, "end": 10,
        "blocks": [{"id": "b", "kind": "paragraph", "evidence_ids": ["k"], "text": "旧正文"},
                   {"id": "s", "kind": "summary", "evidence_ids": ["k"], "text": "旧小结"}]}]}


def test_review_resumes_after_one_of_two_repairs(tmp_path):
    issues = {"issues": [{"block_id": "b", "severity": "error", "reason": "错误一"},
                         {"block_id": "s", "severity": "error", "reason": "错误二"}]}
    with pytest.raises(RuntimeError):
        review(original(), [], Chat([issues, "新正文", RuntimeError("interrupted")]), "m", directory=tmp_path)
    resumed = original()
    client = Chat(["新小结", {"issues": []}])
    result = review(resumed, [], client, "m", directory=tmp_path)
    assert result["status"] == "complete"
    assert [b["text"] for b in resumed["chapters"][0]["blocks"]] == ["新正文", "新小结"]
    assert len(client.calls) == 2
    again = original()
    review(again, [], Chat(), "m", directory=tmp_path)
    assert again == resumed


def test_review_inserts_missing_evidence_block(tmp_path):
    document = original()
    client = Chat([
        {"issues": [{"block_id": None, "severity": "error", "dimension": "coverage",
                     "action": "insert_blocks", "reason": "遗漏前提"}]},
        {"mode": "insert", "blocks": [{"kind": "paragraph", "title": "前提", "evidence_ids": ["k"]}]},
        "补充的前提", {"issues": []},
    ])
    review(document, [], client, "m", directory=tmp_path)
    blocks = document["chapters"][0]["blocks"]
    assert [b["text"] for b in blocks] == ["旧正文", "补充的前提", "旧小结"]
    assert document["outline_revisions"]


def test_structural_repair_resumes_midway_through_new_blocks(tmp_path):
    issue = {"issues": [{"block_id": None, "severity": "error", "action": "insert_blocks", "reason": "缺两步"}]}
    plan = {"mode": "insert", "blocks": [{"kind": "paragraph", "evidence_ids": ["k"]},
                                         {"kind": "paragraph", "evidence_ids": ["k"]}]}
    with pytest.raises(RuntimeError):
        review(original(), [], Chat([issue, plan, "第一步", RuntimeError("interrupt")]), "m", directory=tmp_path)
    document = original()
    client = Chat(["第二步", {"issues": []}])
    review(document, [], client, "m", directory=tmp_path)
    assert [b["text"] for b in document["chapters"][0]["blocks"]] == ["旧正文", "第一步", "第二步", "旧小结"]
    assert len(client.calls) == 2


def test_review_replans_chapter_and_keeps_summary(tmp_path):
    document = original()
    client = Chat([
        {"issues": [{"block_id": "b", "severity": "error", "action": "replan_chapter", "reason": "顺序错误"}]},
        {"mode": "replace", "title": "重组章节", "blocks": [
            {"kind": "paragraph", "title": "机制", "evidence_ids": ["k"]},
            {"kind": "summary", "title": "小结", "evidence_ids": ["k"]}]},
        "先讲机制", "新小结", {"issues": []},
    ])
    review(document, [], client, "m", directory=tmp_path)
    assert document["chapters"][0]["title"] == "重组章节"
    assert [b["text"] for b in document["chapters"][0]["blocks"]] == ["先讲机制", "新小结"]


def test_visual_failure_reselects_and_reaudits_then_restores_cache(tmp_path):
    a, b = tmp_path / "old.jpg", tmp_path / "new.jpg"
    a.write_bytes(b"old")
    b.write_bytes(b"new")
    document = original()
    document["figures"] = [{"figure_id": "f", "request_id": "r", "path": str(a), "start": 0, "end": 10, "timestamp": 2}]
    document["chapters"][0]["blocks"].insert(1, {"id": "image", "kind": "figure", "figure_id": "f"})
    initial = deepcopy(document)
    calls = []
    def visual(local):
        return [{"block_id": "image", "severity": "error", "action": "reselect_figure", "reason": "图例缺失"}] if local["figures"][0]["path"] == str(a) else []
    def repair(local, issue, number):
        calls.append(issue)
        local["figures"][0]["path"] = str(b)
        return True
    result = review(document, [], Chat([{"issues": []}, {"issues": []}]), "m", directory=tmp_path / "review",
                    visual_check=visual, figure_repair=repair)
    assert result["status"] == "complete" and len(calls) == 1
    review(initial, [], Chat(), "m", directory=tmp_path / "review", visual_check=visual, figure_repair=repair)
    assert initial["figures"][0]["path"] == str(b) and len(calls) == 1


def test_failed_visual_repair_cannot_be_silenced_by_next_text_audit(tmp_path):
    client = Chat([{"issues": [{"block_id": "missing", "request_id": "r", "severity": "error",
                              "action": "reselect_figure", "reason": "图缺失"}]}, {"issues": []}])
    result = review(original(), [], client, "m", rounds=1, directory=tmp_path,
                    figure_repair=lambda local, issue, attempt: False)
    assert result["status"] == "partial" and result["issues"][0]["reason"] == "图缺失"


def test_review_only_rechecks_changed_chapter(tmp_path):
    document = original()
    second = deepcopy(document["chapters"][0])
    second.update(title="第二章", start=10, end=20)
    second["blocks"] = [{"id": "s2", "kind": "summary", "text": "第二章内容"}]
    document["chapters"].append(second)
    units = [EvidenceUnit("u1", 0, 10, "第一章"), EvidenceUnit("u2", 10, 20, "第二章")]
    review(document, units, Chat([{"issues": []}, {"issues": []}]), "m", directory=tmp_path)
    document["chapters"][0]["blocks"][0]["text"] = "修改第一章"
    client = Chat([{"issues": []}])
    review(document, units, client, "m", directory=tmp_path)
    assert len(client.calls) == 1


def test_compose_resumes_completed_block_without_replanning(tmp_path):
    chapters = [Chapter("课程", 0, 10, "初稿")]
    units = [EvidenceUnit("u", 0, 10, "原文")]
    plan = {"blocks": [{"kind": "paragraph", "title": "正文"}, {"kind": "summary", "title": "小结"}]}
    with pytest.raises(RuntimeError):
        compose(chapters, units, [], [], Chat([plan, "已完成正文", RuntimeError("interrupted")]), "m", tmp_path)
    client = Chat(["继续小结"])
    document = compose(chapters, units, [], [], client, "m", tmp_path)
    assert [b["text"] for b in document["chapters"][0]["blocks"]] == ["已完成正文", "继续小结"]
    assert len(client.calls) == 1


def test_image_change_invalidates_only_associated_prose(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"old")
    figure = {"figure_id": "f", "request_id": "r", "path": str(image), "timestamp": 2, "start": 0, "end": 10}
    chapters = [Chapter("第一章", 0, 10, "初稿一"), Chapter("第二章", 10, 20, "初稿二")]
    units = [EvidenceUnit("u1", 0, 10, "原文一"), EvidenceUnit("u2", 10, 20, "原文二")]
    plan = {"blocks": [{"kind": "paragraph", "title": "不依赖图片"},
                       {"kind": "paragraph", "title": "解释图片"},
                       {"kind": "figure", "figure_id": "f"}, {"kind": "summary"}]}
    compose(chapters, units, [], [figure], Chat([plan, "不变", "图片原解", "原小结",
        {"blocks": [{"kind": "summary"}]}, "第二章小结"]), "m", tmp_path / "cache")
    image.write_bytes(b"new")
    client = Chat(["图片新解", "更新小结"])
    result = compose(chapters, units, [], [figure], client, "m", tmp_path / "cache")
    assert len(client.calls) == 2
    assert result["chapters"][0]["blocks"][0]["text"] == "不变"
    assert result["chapters"][1]["blocks"][0]["text"] == "第二章小结"


def test_image_change_preserves_unrelated_completed_text_repair(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"old")
    document = original()
    document["figures"] = [{"figure_id": "f", "path": str(image), "start": 0, "end": 10, "timestamp": 5}]
    document["chapters"][0]["blocks"].insert(1, {"id": "explanation", "kind": "paragraph", "text": "图解"})
    document["chapters"][0]["blocks"].insert(2, {"id": "image", "kind": "figure", "figure_id": "f"})
    baseline = deepcopy(document)
    review(document, [], Chat([{"issues": [{"block_id": "b", "severity": "error", "reason": "背景错误"}]},
                               "修正背景", {"issues": []}]), "m", directory=tmp_path / "cache")
    image.write_bytes(b"new")
    client = Chat([{"issues": []}])
    review(baseline, [], client, "m", directory=tmp_path / "cache")
    assert baseline["chapters"][0]["blocks"][0]["text"] == "修正背景"
    assert len(client.calls) == 1
