import json

import pytest

from bbvs.authoring import build_evidence_units, map_content, plan_outline, select_requested_frames
from bbvs.models import ContentMap, Frame, Transcript, TranscriptSegment


class FakeChat:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, **kwargs):
        return json.dumps(next(self.responses), ensure_ascii=False)


def transcript() -> Transcript:
    return Transcript("zh", 240, [
        TranscriptSegment(0, 100, "先解释问题"),
        TranscriptSegment(100, 150, "现在看公式"),
        TranscriptSegment(150, 240, "最后给出结论"),
    ], "manual-subtitle", "platform")


def test_evidence_windows_are_not_fixed_chapters_and_preserve_speech() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=120)
    assert [(row.unit_id, row.start, row.end) for row in units] == [
        ("u_0001", 0, 150), ("u_0002", 150, 240),
    ]
    assert "现在看公式" in units[0].speech


def test_content_map_preserves_formula_signal_without_transcription() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    client = FakeChat([{"maps": [{
        "unit_id": "u_0001", "teaching_goal": "解释注意力",
        "claims": ["复杂度随序列增长"], "mechanisms": [], "examples": [],
        "formulas_or_code": ["画面出现复杂度公式"], "limitations": [],
        "visual_requests": [], "uncertainties": [],
    }]}])
    maps = map_content(units, client, "m")
    assert maps[0].formulas_or_code == ["画面出现复杂度公式"]


def test_formula_signal_selects_local_frames_without_global_limit() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=240)
    maps = [ContentMap("u_0001", formulas_or_code=["公式"])]
    frames = [Frame(f"{index}.jpg", index * 10, text=["x"] * index) for index in range(1, 9)]
    selected = select_requested_frames(units, maps, frames)
    assert [row.path for row in selected] == ["6.jpg", "7.jpg", "8.jpg"]


def test_outline_must_cover_units_once_in_order() -> None:
    units = build_evidence_units(transcript(), [], target_seconds=120)
    maps = [ContentMap(row.unit_id) for row in units]
    client = FakeChat([{"chapters": [{
        "title": "只有后半", "start_unit": "u_0002", "end_unit": "u_0002",
        "required_units": ["u_0002"], "visual_requests": [],
    }]}])
    with pytest.raises(ValueError, match="cover all evidence units"):
        plan_outline(units, maps, {}, client, "m")
