from pathlib import Path

import pytest

from bbvs.io import write_json
from bbvs.report import ReportOptions, build_html, export_report


class FakeRenderer:
    def __init__(self): self.called = False
    def render(self, document: str, output: Path, base_url: Path) -> None:
        self.called = True
        assert "课程&lt;测试&gt;" in document
        output.write_bytes(b"%PDF-fake")


def sample_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    write_json(run / "source" / "metadata.json", {
        "title": "课程<测试>", "duration": 65, "webpage_url": "https://example.test/video",
        "tags": ["经济学"],
    })
    write_json(run / "analysis" / "terminology.json", [{
        "canonical_name": "Money Demand", "category": "concept", "confidence": 0.9,
        "evidence": [{"text": "Money Demand"}],
    }])
    return run


def test_partial_html_report_is_self_describing_and_escaped(tmp_path: Path) -> None:
    document = build_html(sample_run(tmp_path), ReportOptions(max_images=0))
    assert "课程&lt;测试&gt;" in document
    assert "尚未运行" in document
    assert "Money Demand" in document


def test_html_and_pdf_use_same_document(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    html_output = tmp_path / "report.html"
    export_report(run, html_output, ReportOptions(max_images=0))
    assert html_output.read_text(encoding="utf-8").startswith("<!doctype html>")
    renderer = FakeRenderer()
    pdf_output = tmp_path / "report.pdf"
    export_report(run, pdf_output, ReportOptions(max_images=0), renderer)
    assert renderer.called and pdf_output.read_bytes() == b"%PDF-fake"


def test_report_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".html 或 .pdf"):
        export_report(sample_run(tmp_path), tmp_path / "report.docx")


def test_report_renders_original_and_chinese_translation(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    write_json(run / "analysis" / "summary.json", {
        "summary": "Money demand falls as rates rise.",
        "summary_zh": "利率上升时，货币需求下降。",
        "key_concepts": ["Money demand"], "key_concepts_zh": ["货币需求"],
        "takeaways": [], "takeaways_zh": [],
    })
    write_json(run / "analysis" / "chapters.json", [{
        "title": "Interest Rates", "title_zh": "利率", "start": 0, "end": 60,
        "summary": "An introduction.", "summary_zh": "介绍。",
        "key_points": ["Rates matter"], "key_points_zh": ["利率很重要"],
    }])
    document = build_html(run, ReportOptions(max_images=0))
    assert "Money demand falls" in document
    assert "利率上升时，货币需求下降。" in document
    assert "Interest Rates" in document and "利率" in document


def test_report_pairs_each_image_with_timestamped_bilingual_summary(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    write_json(run / "ocr-test.json", [{"path": str(frame), "timestamp": 42, "text": ["Money Demand"]}])
    write_json(run / "analysis" / "visual-analysis.json", [{
        "timestamp": 42, "summary": "The slide shows money demand.",
        "summary_zh": "幻灯片展示货币需求。", "visual_type": "slide",
        "entities": [], "relations": [],
    }])
    document = build_html(run, ReportOptions(max_images=1))
    assert "图文时间轴" in document
    assert "The slide shows money demand." in document
    assert "幻灯片展示货币需求。" in document
    assert "00:00:42" in document


def test_basic_report_does_not_warn_about_intentionally_disabled_vision(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    write_json(run / "analysis" / "summary.json", {"summary": "Done"})
    write_json(run / "analysis" / "chapters.json", [{
        "title": "Chapter", "start": 0, "end": 60, "summary": "Done", "key_points": [],
    }])
    write_json(run / "analysis" / "verified-transcript.json", {"segments": []})
    document = build_html(run, ReportOptions(max_images=0, expect_vision=False))
    assert "尚未运行" not in document
    assert "视觉分析" not in document
