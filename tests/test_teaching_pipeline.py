from dataclasses import asdict

from bbvs.authoring import AUTHORING_PROMPT_VERSION
from bbvs.config import Endpoint, Services
from bbvs.io import read_json, write_json
from bbvs.models import Transcript, TranscriptSegment
from bbvs.pipeline import VideoPipeline
from test_teaching_recovery import Chat


def test_pipeline_restores_repaired_document_and_synthesizes_after_repair(tmp_path):
    run = tmp_path / "run"
    analysis = run / "analysis"
    transcript = run / "asr/current/transcript.json"
    frames = run / "ocr/current/frames.json"
    write_json(run / "source/metadata.json", {"title": "课程"})
    write_json(transcript, asdict(Transcript("zh", 10, [TranscriptSegment(0, 10, "结论有前提")], "fake", "fake")))
    write_json(frames, [])
    write_json(analysis / "terminology.json", [])
    write_json(analysis / "authoring-version.json", {"prompt_version": AUTHORING_PROMPT_VERSION})
    write_json(analysis / "evidence-units.json", [{"unit_id": "u_0001", "start": 0, "end": 10,
        "speech": "结论有前提", "screen_text": [], "source_chapter": None}])
    write_json(analysis / "content-map.json", [{"unit_id": "u_0001", "knowledge_items": [{
        "id": "k", "content": "结论有前提", "control": {"form": "proposition", "role": "main", "importance": "essential"}}]}])
    write_json(analysis / "outline.json", {"chapters": [{"title": "课程", "start_unit": "u_0001", "end_unit": "u_0001",
        "topics": [{"title": "结论", "summary": "结论", "required_items": ["k"], "supporting_items": []}]}]})
    pipeline = VideoPipeline(Services(Endpoint("http://unused/v1", "m")))
    pipeline.llm = Chat(["标题：课程\n结论", {"missing_essential": [], "lost_reasoning": [], "distortions": [],
        "unsupported_claims": [], "coherence_issues": []},
        {"blocks": [{"kind": "summary", "title": "小结", "evidence_ids": ["k"]}]}, "只有结论",
        {"issues": [{"block_id": None, "severity": "error", "action": "insert_blocks", "reason": "遗漏前提"}]},
        {"mode": "insert", "blocks": [{"kind": "paragraph", "title": "前提", "evidence_ids": ["k"]}]},
        "补回前提", {"issues": []}, "综合包含前提"])
    pipeline.analyze(run, summarize=True, authoring=True, transcript_path=transcript, frames_path=frames)
    document = read_json(analysis / "teaching-document.json")
    assert document["status"] == "complete"
    assert "补回前提" in pipeline.llm.calls[-1]["messages"][0]["content"]
    assert read_json(analysis / "outline-revisions.json")
    pipeline.llm = Chat()
    pipeline.analyze(run, summarize=True, authoring=True, transcript_path=transcript, frames_path=frames)
    assert pipeline.llm.calls == []
    # Simulate an interruption after review: final files are gone, but chapter/block/checkpoints remain.
    (analysis / "teaching-document.json").unlink()
    pipeline.analyze(run, summarize=True, authoring=True, transcript_path=transcript, frames_path=frames)
    assert pipeline.llm.calls == []
    assert read_json(analysis / "teaching-document.json")["chapters"] == document["chapters"]
