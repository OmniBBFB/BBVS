from __future__ import annotations

from pathlib import Path

from .models import Transcript, TranscriptSegment
from .io import read_json
import re


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


_SRT_TIME = re.compile(
    r"(?P<sh>\d+):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d+):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
)


def _seconds(match: re.Match[str], prefix: str) -> float:
    return (int(match.group(prefix + "h")) * 3600 + int(match.group(prefix + "m")) * 60
            + int(match.group(prefix + "s")) + int(match.group(prefix + "ms")) / 1000)


def from_srt(path: Path, language: str, source: str = "subtitle") -> Transcript:
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((index for index, line in enumerate(lines) if _SRT_TIME.fullmatch(line)), None)
        if time_index is None:
            continue
        match = _SRT_TIME.fullmatch(lines[time_index])
        assert match is not None
        text = " ".join(re.sub(r"<[^>]+>", "", line) for line in lines[time_index + 1:]).strip()
        if text:
            segments.append(TranscriptSegment(_seconds(match, "s"), _seconds(match, "e"), text))
    duration = max((row.end for row in segments), default=0.0)
    return Transcript(language=language, duration=duration, segments=segments, engine=source, model="platform")


def preferred_platform_transcript(run_dir: Path) -> Transcript | None:
    metadata_path = run_dir / "source" / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = read_json(metadata_path)
    tracks = metadata.get("subtitle_tracks") or []
    language_rank = ("zh-Hans", "zh-CN", "zh", "ai-zh", "en")

    source_language = str(metadata.get("language") or "").casefold()

    def rank(row: dict) -> tuple[int, int, int]:
        kind = 0 if row.get("kind") == "manual" else 1
        language = str(row.get("language", ""))
        language_match = 0 if source_language and language.casefold().split("-")[0] == source_language.split("-")[0] else 1
        return language_match, kind, language_rank.index(language) if language in language_rank else len(language_rank)

    for track in sorted((row for row in tracks if isinstance(row, dict)), key=rank):
        filename = Path(str(track.get("filename", ""))).name
        candidates = [run_dir / "source" / filename]
        candidates.extend((run_dir / "source").glob(f"source.{track.get('language')}*.srt"))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path:
            transcript = from_srt(path, str(track.get("language") or "unknown"),
                                  f"{track.get('kind', 'platform')}-subtitle")
            if transcript.segments:
                return transcript
    return None
