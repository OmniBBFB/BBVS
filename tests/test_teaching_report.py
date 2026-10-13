from bbvs.io import write_json
from bbvs.report import ReportOptions, build_html, export_report


def run_fixture(tmp_path):
    run = tmp_path / "run"
    write_json(run / "source/metadata.json", {
        "title": "课程<测试>", "duration": 65, "uploader": "作者", "upload_date": "20261013",
        "tags": ["领域"], "webpage_url": "https://example.test/video"})
    write_json(run / "analysis/teaching-document.json", {
        "status": "complete", "evidence_ids": [], "figures": [], "synthesis": "最终综合",
        "chapters": [{"title": "教学章节", "blocks": [{"id": "s", "kind": "summary", "text": "正文<引用>"}]}]})
    write_json(run / "analysis/terminology.json", [{"canonical_name": "术语<实体>", "category": "领域",
        "confidence": 1, "evidence": [{"text": "来自字幕"}]}])
    write_json(run / "analysis/corrections.json", [{"start": 3, "original": "旧术语", "corrected": "新术语", "confidence": 1}])
    write_json(run / "asr/current/transcript.json", {"segments": [{"start": 4, "text": "原始转录"}]})
    return run


def test_teaching_report_preserves_appendices_metadata_and_transcript_option(tmp_path):
    run = run_fixture(tmp_path)
    document = build_html(run, ReportOptions(include_transcript=True, expect_vision=False))
    assert "教学章节" in document and "正文&lt;引用&gt;" in document
    assert "术语表" in document and "术语&lt;实体&gt;" in document
    assert "转录校正" in document and "新术语" in document
    assert "完整转录" in document and "原始转录" in document
    assert "作者" in document and "20261013" in document and "领域" in document
    assert "原始转录" not in build_html(run, ReportOptions(expect_vision=False))


def test_teaching_report_prefers_verified_and_explicit_transcript(tmp_path):
    run = run_fixture(tmp_path)
    explicit = run / "explicit.json"
    write_json(explicit, {"segments": [{"start": 1, "text": "显式转录"}]})
    result = build_html(run, ReportOptions(include_transcript=True), transcript_path=explicit)
    assert "显式转录" in result and "原始转录" not in result
    write_json(run / "analysis/verified-transcript.json", {"segments": [{"start": 2, "text": "校验转录"}]})
    result = build_html(run, ReportOptions(include_transcript=True), transcript_path=explicit)
    assert "校验转录" in result and "显式转录" not in result


def test_teaching_html_and_pdf_share_document_and_escaped_partial_details(tmp_path):
    run = run_fixture(tmp_path)
    doc_path = run / "analysis/teaching-document.json"
    from bbvs.io import read_json
    doc = read_json(doc_path)
    doc.update(status="partial", unresolved=[{"reason": "图缺失<script>"}])
    write_json(doc_path, doc)
    write_json(run / "analysis/document-review.json", {"issues": [{"reason": "前提缺失"}]})
    output = run / "notes.html"
    options = ReportOptions(include_transcript=True)
    export_report(run, output, options)
    html = output.read_text()
    class Renderer:
        def render(self, document, target, base_url):
            assert document == html
            target.write_bytes(b"%PDF-test")
    export_report(run, run / "notes.pdf", options, Renderer())
    assert "图缺失&lt;script&gt;" in html and "前提缺失" in html
    assert "原始转录" in html


def test_tex_export_preserves_appendices_and_transcript_option(tmp_path):
    run = run_fixture(tmp_path)
    output = run / "notes.tex"
    export_report(run, output, ReportOptions(include_transcript=True))
    text = output.read_text()
    assert "术语表" in text and "新术语" in text and "原始转录" in text
    export_report(run, output, ReportOptions(include_transcript=False))
    assert "完整转录" not in output.read_text()


def test_report_stage_invalidates_export_when_document_changes(tmp_path, monkeypatch):
    from bbvs.config import Endpoint, Services
    from bbvs.runner import StageContext, ReportStage
    from bbvs.settings import AppSettings, ReportSettings, RetrySettings
    from bbvs.io import read_json
    import bbvs.runner as runner
    run = run_fixture(tmp_path)
    settings = AppSettings(Services(Endpoint("http://unused/v1", "m")),
                           report=ReportSettings(filename="notes.html"))
    context = StageContext("BV1Rch76WEfQ", settings, progress=lambda message: None,
        run_dir=run, analysis_dir=run / "analysis", asr_path=run / "asr/current/transcript.json",
        ocr_path=run / "ocr/current/frames.json")
    calls = []
    def export(*args, **kwargs):
        calls.append(args[1])
        return export_report(*args, **kwargs)
    monkeypatch.setattr(runner, "export_report", export)
    stage = ReportStage(RetrySettings())
    stage.run(context)
    stage.run(context)
    assert len(calls) == 1
    path = run / "analysis/teaching-document.json"
    document = read_json(path)
    document["chapters"][0]["blocks"][0]["text"] = "修改后的正文"
    write_json(path, document)
    stage.run(context)
    assert len(calls) == 2
    output = next(calls[-1].parent.glob("notes.html"))
    assert "修改后的正文" in output.read_text()
