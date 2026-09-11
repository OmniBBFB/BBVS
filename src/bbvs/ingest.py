from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .io import write_json
from .process import require_executable, run
from .archive import archive_run


def download(
    url: str, output_dir: Path, *, cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    yt_dlp = require_executable("yt-dlp")
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "source.%(ext)s")
    args = [
            yt_dlp,
            "--no-playlist",
            "--write-info-json",
            "--write-subs",
            "--write-auto-subs",
            "--convert-subs",
            "srt",
            "--sub-langs",
            "all,-live_chat,-danmaku",
            "--merge-output-format",
            "mp4",
            "--print-json",
            "-o",
            template,
            url,
        ]
    if cookies_from_browser:
        args[1:1] = ["--cookies-from-browser", cookies_from_browser]
    elif cookies_file:
        args[1:1] = ["--cookies", str(cookies_file)]
    result = run(args)
    metadata = json.loads(result.stdout.strip().splitlines()[-1])
    requested = metadata.get("requested_downloads") or []
    candidates = [Path(item["filepath"]) for item in requested if item.get("filepath")]
    candidates.extend(output_dir.glob("source.*"))
    media_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
    media = next(
        (
            path
            for path in candidates
            if path.exists() and path.suffix.lower() in media_extensions
        ),
        None,
    )
    if media is None:
        raise RuntimeError("yt-dlp 已完成，但无法定位下载的视频文件")
    compact = {
        key: metadata.get(key)
        for key in (
            "id",
            "webpage_url",
            "title",
            "description",
            "tags",
            "chapters",
            "duration",
            "uploader",
            "upload_date",
            "language",
        )
    }
    manual_languages = set((metadata.get("subtitles") or {}).keys())
    automatic_languages = set((metadata.get("automatic_captions") or {}).keys())
    requested_subtitles = metadata.get("requested_subtitles") or {}
    compact["subtitle_tracks"] = [
        {
            "language": language,
            "kind": "manual" if language in manual_languages else "automatic",
            "filename": Path(str(details.get("filepath", f"source.{language}.srt"))).name,
        }
        for language, details in requested_subtitles.items()
        if language in manual_languages or language in automatic_languages
    ]
    write_json(output_dir / "metadata.json", compact)
    return media.resolve(), compact


def download_to_run(
    url: str, runs_dir: Path, *, cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=runs_dir))
    try:
        media, metadata = download(
            url, pending / "source", cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
        run_dir = archive_run(pending, runs_dir)
        media = run_dir / media.relative_to(pending)
        return run_dir, media, metadata
    except Exception:
        # Keep the pending directory for diagnosing or resuming a failed download.
        raise
