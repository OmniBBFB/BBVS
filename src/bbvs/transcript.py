from __future__ import annotations

from pathlib import Path

from .models import Transcript, TranscriptSegment


def _srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for index, segment in enumerate(transcript.segments, start=1):
        blocks.append(
            f"{index}\n{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}\n"
            f"{segment.text}"
        )
    return "\n\n".join(blocks) + "\n"


def to_text(transcript: Transcript) -> str:
    return "\n".join(segment.text for segment in transcript.segments) + "\n"


def from_dict(payload: dict) -> Transcript:
    return Transcript(
        language=payload.get("language"),
        duration=payload.get("duration"),
        segments=[TranscriptSegment(**segment) for segment in payload.get("segments", [])],
        engine=payload["engine"],
        model=payload["model"],
    )


def export(transcript: Transcript, output: Path, format_name: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    content = to_srt(transcript) if format_name == "srt" else to_text(transcript)
    output.write_text(content, encoding="utf-8")
    return output

