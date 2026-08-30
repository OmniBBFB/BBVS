from bbvs.media import _number


def test_number_handles_ffprobe_na() -> None:
    assert _number("12.5") == 12.5
    assert _number("N/A") is None
    assert _number(None) is None
