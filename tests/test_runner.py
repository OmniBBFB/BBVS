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
        lambda path, output, options: events.append(("report", output)) or output,
    )
    settings = AppSettings(services=Services(Endpoint("http://llm/v1", "m")), runs_dir=tmp_path / "runs")

    result = run_pipeline("https://video", settings, progress=lambda message: events.append(message))

    assert result == run_dir
    assert (run_dir / "audio.wav").exists()
    assert (run_dir / "keyframes" / "keyframes.json").exists()
    assert (run_dir / "ocr-rapidocr.json").exists()
    assert (run_dir / "asr-faster-whisper-small.json").exists()
    assert any(isinstance(event, tuple) and event[0] == "analyze" for event in events)
    assert any(isinstance(event, tuple) and event[0] == "report" for event in events)
