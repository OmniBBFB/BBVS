from pathlib import Path

from bbvs.config import Endpoint, Services
from bbvs.models import Frame, Transcript, TranscriptSegment
from bbvs.runner import run_pipeline
from bbvs.settings import AppSettings


def test_one_command_runner_executes_all_stages(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "BV1-Title"
    video = run_dir / "source" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    events = []

    monkeypatch.setattr("bbvs.runner.ingest.download_to_run", lambda source, root: (run_dir, video, {}))
    monkeypatch.setattr("bbvs.runner.media.extract_audio", lambda source, output: output.write_bytes(b"wav") or output)
    monkeypatch.setattr(
        "bbvs.runner.keyframes.extract_keyframes",
        lambda source, output, threshold, maximum: [Frame(str(output / "frame.jpg"), 0)],
    )
    monkeypatch.setattr("bbvs.runner.ocr.recognize", lambda frames, options, engine: frames)
    monkeypatch.setattr(
        "bbvs.runner.asr.transcribe",
        lambda audio, options, engine: Transcript("en", 1, [TranscriptSegment(0, 1, "hello")], engine, options.model),
    )

    class FakePipeline:
        def __init__(self, services): pass
        def analyze(self, path, **options):
            events.append(("analyze", options))

    monkeypatch.setattr("bbvs.runner.VideoPipeline", FakePipeline)
    monkeypatch.setattr(
        "bbvs.runner.export_report",
        lambda path, output, options, **inputs: events.append(("report", output)) or output,
    )
    settings = AppSettings(services=Services(Endpoint("http://llm/v1", "m")), runs_dir=tmp_path / "runs")

    result = run_pipeline("https://video", settings, progress=lambda message: events.append(message))

    assert result == run_dir
    assert (run_dir / "audio" / "audio.wav").exists()
    assert len(list((run_dir / "keyframes").glob("*/keyframes.json"))) == 1
    assert len(list((run_dir / "ocr").glob("*/frames.json"))) == 1
    assert len(list((run_dir / "asr").glob("*/transcript.json"))) == 1
    analyze_event = next(event for event in events if isinstance(event, tuple) and event[0] == "analyze")
    assert analyze_event[1]["analysis_dir"].parent == run_dir / "analysis"
    report_event = next(event for event in events if isinstance(event, tuple) and event[0] == "report")
    assert report_event[1].parent.parent == run_dir / "reports"
    assert any(isinstance(event, tuple) and event[0] == "analyze" for event in events)
    assert any(isinstance(event, tuple) and event[0] == "report" for event in events)
    elapsed = [event for event in events if isinstance(event, str) and "完成，耗时" in event]
    assert len(elapsed) == 7
    assert all(event.endswith(" 秒") for event in elapsed)
