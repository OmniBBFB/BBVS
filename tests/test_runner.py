from pathlib import Path

import pytest

from bbvs.config import Endpoint, Services
from bbvs.errors import BBVSError
from bbvs.models import Frame, Transcript, TranscriptSegment
from bbvs.io import read_json
from bbvs.runner import (
    AnalysisStage, PipelineStage, StageContext, _batch_run_dir, _manifest_sources, _resolve_run_dir,
    run_batch, run_pipeline, run_source,
)
from bbvs.settings import AppSettings, RetrySettings


def test_resolve_run_dir_accepts_name_relative_to_configured_runs_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    expected = runs_dir / "BV1-标题"
    expected.mkdir(parents=True)

    assert _resolve_run_dir("BV1-标题", runs_dir) == expected.resolve()


def test_manifest_parses_bvids_urls_comments_and_duplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "videos.txt"
    manifest.write_text(
        "# season\nBV1E1xxebEDs\n\nhttps://www.bilibili.com/video/BV1Nh1BYKEEA?p=1\n"
        "BV1E1xxebEDs\n",
        encoding="utf-8",
    )

    assert _manifest_sources(manifest) == [
        ("BV1E1xxebEDs", "https://www.bilibili.com/video/BV1E1xxebEDs"),
        ("BV1Nh1BYKEEA", "https://www.bilibili.com/video/BV1Nh1BYKEEA"),
    ]


def test_batch_continues_after_video_failure_and_saves_progress(tmp_path: Path) -> None:
    manifest = tmp_path / "season.txt"
    manifest.write_text("BV1E1xxebEDs\nBV1Nh1BYKEEA\n", encoding="utf-8")
    settings = AppSettings(
        services=Services(Endpoint("http://llm/v1", "m")), runs_dir=tmp_path / "runs",
    )
    calls = []

    def fake_runner(source, settings, progress, sleeper):
        calls.append(source)
        if source.endswith("BV1E1xxebEDs"):
            raise BBVSError("model unavailable")
        return settings.runs_dir / "BV1Nh1BYKEEA-title"

    with pytest.raises(BBVSError, match="1 个视频失败"):
        run_batch(manifest, settings, progress=lambda message: None, runner=fake_runner)

    assert len(calls) == 2
    payload = read_json(settings.runs_dir / "batches" / "season.json")
    assert [item["status"] for item in payload["items"]] == ["failed", "completed"]


def test_batch_reuses_run_directory_matching_bvid(tmp_path: Path) -> None:
    manifest = tmp_path / "season.txt"
    manifest.write_text("BV1E1xxebEDs\n", encoding="utf-8")
    settings = AppSettings(
        services=Services(Endpoint("http://llm/v1", "m")), runs_dir=tmp_path / "runs",
    )
    existing = settings.runs_dir / "BV1E1xxebEDs-title"
    existing.mkdir(parents=True)
    received = []

    def fake_runner(source, settings, progress, sleeper):
        received.append(source)
        return Path(source)

    run_batch(manifest, settings, progress=lambda message: None, runner=fake_runner)

    assert received == [str(existing.resolve())]
    assert _batch_run_dir(settings.runs_dir, "BV1E1xxebEDs") == existing.resolve()


def test_run_source_treats_txt_as_batch_manifest(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "season.txt"
    manifest.write_text("BV1E1xxebEDs\n", encoding="utf-8")
    expected = tmp_path / "result.json"
    monkeypatch.setattr("bbvs.runner.run_batch", lambda *args, **kwargs: expected)
    settings = AppSettings(services=Services(Endpoint("http://llm/v1", "m")))

    assert run_source(str(manifest), settings) == expected


def test_one_command_runner_executes_all_stages(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "BV1-Title"
    video = run_dir / "source" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    events = []

    monkeypatch.setattr(
        "bbvs.runner.ingest.download_to_run",
        lambda source, root, **options: (run_dir, video, {}),
    )
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
        lambda path, output, options, **inputs: (
            events.append(("report", output)), output.parent.mkdir(parents=True, exist_ok=True),
            output.write_bytes(b"pdf"), output,
        )[-1],
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


def test_stage_retries_transient_failure_with_exponential_backoff(tmp_path: Path) -> None:
    attempts = []
    delays = []
    events = []

    class FlakyStage(PipelineStage):
        number, title = 1, "flaky"

        def _run(self, context):
            attempts.append(1)
            if len(attempts) < 3:
                raise BBVSError("temporary")
            context.run_dir = tmp_path

    context = StageContext(
        "source", AppSettings(services=Services(Endpoint("http://llm/v1", "m"))),
        progress=events.append,
    )
    FlakyStage(
        RetrySettings(attempts=3, initial_delay=2, multiplier=2, max_delay=30),
        sleeper=delays.append,
    ).run(context)

    assert len(attempts) == 3
    assert delays == [2, 4]
    assert sum("次尝试失败" in event for event in events) == 2


def test_stage_does_not_retry_permanent_configuration_failure(tmp_path: Path) -> None:
    attempts = []
    delays = []

    class InvalidStage(PipelineStage):
        number, title = 1, "invalid"

        def _run(self, context):
            attempts.append(1)
            raise ValueError("invalid config")

    context = StageContext(
        "source", AppSettings(services=Services(Endpoint("http://llm/v1", "m"))),
        progress=lambda message: None,
    )
    with pytest.raises(ValueError, match="invalid config"):
        InvalidStage(RetrySettings(), sleeper=delays.append).run(context)

    assert len(attempts) == 1
    assert delays == []


def test_analysis_retry_recreates_remote_client(monkeypatch, tmp_path: Path) -> None:
    clients = []
    delays = []

    class FakePipeline:
        def __init__(self, services):
            clients.append(self)

        def analyze(self, path, **options):
            if len(clients) == 1:
                raise BBVSError("empty model content")

    monkeypatch.setattr("bbvs.runner.VideoPipeline", FakePipeline)
    settings = AppSettings(services=Services(Endpoint("http://llm/v1", "m")))
    context = StageContext(
        "source", settings, progress=lambda message: None, run_dir=tmp_path,
        asr_dir=tmp_path / "asr" / "variant", asr_path=tmp_path / "asr" / "transcript.json",
        ocr_path=tmp_path / "ocr" / "variant" / "frames.json",
    )

    AnalysisStage(RetrySettings(), sleeper=delays.append).run(context)

    assert len(clients) == 2
    assert delays == [2]
    assert context.analysis_dir is not None
