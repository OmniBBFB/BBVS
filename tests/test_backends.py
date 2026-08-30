from pathlib import Path

import pytest

from bbvs import asr, ocr
from bbvs.models import Frame, Transcript


class FakeASR:
    name = "test-fake-asr"

    def transcribe(self, audio: Path, options: asr.ASROptions) -> Transcript:
        return Transcript("zh", 1.0, [], self.name, options.model)


class FakeOCR:
    name = "test-fake-ocr"

    def recognize(self, frames: list[Frame], options: ocr.OCROptions) -> list[Frame]:
        frames[0].text = [options.language]
        return frames


def test_external_asr_adapter_uses_common_interface() -> None:
    asr.register_engine(FakeASR.name, FakeASR, replace=True)
    result = asr.transcribe(Path("unused.wav"), asr.ASROptions(model="fake"), FakeASR.name)
    assert result.engine == FakeASR.name
    assert result.model == "fake"


def test_external_ocr_adapter_uses_common_interface() -> None:
    ocr.register_engine(FakeOCR.name, FakeOCR, replace=True)
    result = ocr.recognize([Frame("unused.jpg", 0.0)], ocr.OCROptions("en"), FakeOCR.name)
    assert result[0].text == ["en"]


def test_duplicate_adapter_requires_explicit_replace() -> None:
    with pytest.raises(ValueError, match="已存在"):
        asr.register_engine("faster-whisper", FakeASR)


def test_paddle_v2_and_v3_results_are_normalized() -> None:
    v2 = [[[[0, 0]], ("FlashInfer", 0.99)]]
    v3 = {"res": {"rec_texts": ["vLLM", "PagedAttention"]}}
    assert ocr._paddle_texts(v2) == ["FlashInfer"]
    assert ocr._paddle_texts(v3) == ["vLLM", "PagedAttention"]
