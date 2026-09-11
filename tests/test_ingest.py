import json
from types import SimpleNamespace

from bbvs.ingest import download


def test_download_excludes_bilibili_danmaku_from_subtitle_conversion(monkeypatch, tmp_path) -> None:
    captured = []

    def fake_run(args):
        captured.extend(args)
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({
            "id": "BV1", "title": "demo", "requested_downloads": [],
            "subtitles": {}, "automatic_captions": {}, "requested_subtitles": {},
        }))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)

    download("https://www.bilibili.com/video/BV1", tmp_path)

    languages = captured[captured.index("--sub-langs") + 1]
    assert "-danmaku" in languages.split(",")


def test_download_passes_browser_cookies_to_yt_dlp(monkeypatch, tmp_path) -> None:
    captured = []

    def fake_run(args):
        captured.extend(args)
        (tmp_path / "source.mp4").write_bytes(b"video")
        return SimpleNamespace(stdout=json.dumps({"id": "BV1", "requested_downloads": []}))

    monkeypatch.setattr("bbvs.ingest.require_executable", lambda name: name)
    monkeypatch.setattr("bbvs.ingest.run", fake_run)

    download("https://www.bilibili.com/video/BV1", tmp_path, cookies_from_browser="chrome")

    assert captured[captured.index("--cookies-from-browser") + 1] == "chrome"
