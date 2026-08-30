from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .io import write_json
from .process import require_executable, run
from .archive import archive_run


def download(url: str, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    yt_dlp = require_executable("yt-dlp")
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "source.%(ext)s")
    result = run(
        [
            yt_dlp,
            "--no-playlist",
            "--write-info-json",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            "all,-live_chat",
            "--merge-output-format",
            "mp4",
            "--print-json",
            "-o",
            template,
            url,
        ]
    )
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
        )
    }
    write_json(output_dir / "metadata.json", compact)
    return media.resolve(), compact


def download_to_run(url: str, runs_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=runs_dir))
    try:
        media, metadata = download(url, pending / "source")
        run_dir = archive_run(pending, runs_dir)
        media = run_dir / media.relative_to(pending)
        return run_dir, media, metadata
    except Exception:
        # Keep the pending directory for diagnosing or resuming a failed download.
        raise
