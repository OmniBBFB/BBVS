from bbvs.models import Transcript, TranscriptSegment
from bbvs.transcript import to_srt, to_text


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
