from __future__ import annotations

from .models import Frame, Term, TimelineSegment, Transcript, VisualAnalysis


def build_timeline(
    transcript: Transcript, frames: list[Frame], terms: list[Term],
    visuals: list[VisualAnalysis] | None = None, window_seconds: float = 60.0,
) -> list[TimelineSegment]:
    visuals = visuals or []
    duration = transcript.duration or (transcript.segments[-1].end if transcript.segments else 0.0)
    result = []
    start = 0.0
    names = [term.canonical_name for term in terms]
    while start < duration:
        end = min(duration, start + window_seconds)
        speech = " ".join(row.text for row in transcript.segments if row.end > start and row.start < end)
        window_frames = [frame for frame in frames if start <= frame.timestamp < end]
        screen_text = list(dict.fromkeys(text for frame in window_frames for text in frame.text))
        window_visuals = [item for item in visuals if start <= item.timestamp < end]
        haystack = (speech + " " + " ".join(screen_text)).casefold()
        result.append(TimelineSegment(
            start=start, end=end, speech=speech, screen_text=screen_text,
            visuals=window_visuals, terms=[name for name in names if name.casefold() in haystack],
        ))
        start = end
    return result
