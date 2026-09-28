import json
from types import SimpleNamespace

import pytest

from bbvs.ingest import download
from bbvs.models import MediaInfo


def media_info(path, duration: float | None = 30.0) -> MediaInfo:
    return MediaInfo(
        path=str(path), duration=duration, width=1920, height=1080, fps=30.0,
        has_audio=True, format_name="mp4",
    )


def test_download_excludes_bilibili_danmaku_from_subtitle_conversion(monkeypatch, tmp_path) -> None:
    captured = []

    def fake_run(args):
        captured.extend(args)
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({
            "id": "BV1", "title": "demo", "duration": 30,
            "requested_downloads": [],
            "subtitles": {}, "automatic_captions": {}, "requested_subtitles": {},
        }))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)
    monkeypatch.setattr("bbvs.ingest.probe", lambda path: media_info(path))

    download("https://www.bilibili.com/video/BV1", tmp_path)

    languages = captured[captured.index("--sub-langs") + 1]
    assert "-danmaku" in languages.split(",")


def test_download_passes_browser_cookies_to_yt_dlp(monkeypatch, tmp_path) -> None:
    captured = []

    def fake_run(args):
        captured.extend(args)
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({
            "id": "BV1", "duration": 30, "requested_downloads": [],
        }))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)
    monkeypatch.setattr("bbvs.ingest.probe", lambda path: media_info(path))

    download("https://www.bilibili.com/video/BV1", tmp_path, cookies_from_browser="chrome")

    assert captured[captured.index("--cookies-from-browser") + 1] == "chrome"


def test_download_rejects_media_shorter_than_metadata(monkeypatch, tmp_path) -> None:
    def fake_run(args):
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({
            "id": "BV1", "duration": 431.706, "requested_downloads": [],
            "subtitles": {}, "automatic_captions": {}, "requested_subtitles": {},
        }))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)
    monkeypatch.setattr("bbvs.ingest.probe", lambda path: media_info(path, 90.1))

    with pytest.raises(RuntimeError, match=r"431\.706.*90\.100"):
        download("https://www.bilibili.com/video/BV1", tmp_path)


def test_download_accepts_container_duration_rounding(monkeypatch, tmp_path) -> None:
    def fake_run(args):
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({
            "id": "BV1", "duration": 431.706, "requested_downloads": [],
        }))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)
    monkeypatch.setattr("bbvs.ingest.probe", lambda path: media_info(path, 431.7))

    _, metadata = download("https://www.bilibili.com/video/BV1", tmp_path)

    assert metadata["duration"] == 431.706


@pytest.mark.parametrize(
    ("metadata_duration", "actual_duration", "message"),
    [
        (None, 30.0, "元数据缺少有效的视频时长"),
        (30.0, None, "ffprobe 无法读取下载视频的真实时长"),
    ],
)
def test_download_requires_comparable_durations(
    monkeypatch, tmp_path, metadata_duration, actual_duration, message,
) -> None:
    def fake_run(args):
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({
            "id": "BV1", "duration": metadata_duration, "requested_downloads": [],
        }))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)
    monkeypatch.setattr(
        "bbvs.ingest.probe", lambda path: media_info(path, actual_duration),
    )

    with pytest.raises(RuntimeError, match=message):
        download("https://www.bilibili.com/video/BV1", tmp_path)
