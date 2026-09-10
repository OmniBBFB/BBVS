from bbvs.models import Transcript, TranscriptSegment
from bbvs.io import write_json
from bbvs.transcript import preferred_platform_transcript, to_srt, to_text


def test_transcript_exports_srt_and_text() -> None:
    transcript = Transcript(
        language="en",
        duration=2.0,
        segments=[TranscriptSegment(0.0, 1.2346, "Hello")],
        engine="fake",
        model="fake",
    )
    assert to_srt(transcript) == "1\n00:00:00,000 --> 00:00:01,235\nHello\n"
    assert to_text(transcript) == "Hello\n"


def test_platform_transcript_prefers_manual_subtitle(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "metadata.json", {"subtitle_tracks": [
        {"language": "zh", "kind": "automatic", "filename": "source.zh-auto.srt"},
        {"language": "en", "kind": "manual", "filename": "source.en.srt"},
    ]})
    body = "1\n00:00:00,000 --> 00:00:01,500\n{text}\n"
    (source / "source.zh-auto.srt").write_text(body.format(text="自动字幕"), encoding="utf-8")
    (source / "source.en.srt").write_text(body.format(text="manual"), encoding="utf-8")
    result = preferred_platform_transcript(tmp_path)
    assert result is not None
    assert result.engine == "manual-subtitle"
    assert result.segments[0].text == "manual"
