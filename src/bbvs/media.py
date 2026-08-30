from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from .models import MediaInfo
from .process import require_executable, run


def _number(value: str | None) -> float | None:
    if not value or value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def probe(path: Path) -> MediaInfo:
    ffprobe = require_executable("ffprobe")
    result = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    fps = None
    rate = video.get("avg_frame_rate")
    if rate and rate != "0/0":
        fps = float(Fraction(rate))
    format_info = payload.get("format", {})
    return MediaInfo(
        path=str(path.resolve()),
        duration=_number(format_info.get("duration")),
        width=video.get("width"),
        height=video.get("height"),
        fps=fps,
        has_audio=has_audio,
        format_name=format_info.get("format_name"),
    )


def extract_audio(video: Path, output: Path, sample_rate: int = 16_000) -> Path:
    ffmpeg = require_executable("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    return output

