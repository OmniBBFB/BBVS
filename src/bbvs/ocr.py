from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import DependencyError
from .models import Frame


@dataclass(slots=True)
class OCROptions:
    """Engine-neutral OCR settings exposed to callers."""

    language: str = "ch"


class OCREngine(Protocol):
    name: str

    def recognize(self, frames: list[Frame], options: OCROptions) -> list[Frame]: ...


class RapidOCREngine:
    name = "rapidocr"

    def recognize(self, frames: list[Frame], options: OCROptions) -> list[Frame]:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise DependencyError("缺少 RapidOCR，请运行: uv sync --extra ocr-rapidocr") from exc

        engine = RapidOCR()
        for frame in frames:
            result, _ = engine(frame.path)
            frame.text = [str(item[1]).strip() for item in (result or []) if str(item[1]).strip()]
        return frames


def _paddle_texts(result: Any) -> list[str]:
    """Normalize PaddleOCR 2.x and 3.x result shapes behind the adapter."""
    if result is None:
        return []
    if isinstance(result, dict):
        data = result.get("res", result)
        texts = data.get("rec_texts") if isinstance(data, dict) else None
        return [str(text).strip() for text in (texts or []) if str(text).strip()]
    json_value = getattr(result, "json", None)
    if json_value is not None:
        return _paddle_texts(json_value() if callable(json_value) else json_value)
    if isinstance(result, (list, tuple)):
        texts: list[str] = []
        for item in result:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and item[1]
                and isinstance(item[1][0], str)
            ):
                texts.append(item[1][0].strip())
            else:
                texts.extend(_paddle_texts(item))
        return [text for text in texts if text]
    return []


class PaddleOCREngine:
    name = "paddleocr"

    def recognize(self, frames: list[Frame], options: OCROptions) -> list[Frame]:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise DependencyError("缺少 PaddleOCR，请运行: uv sync --extra ocr-paddleocr") from exc

        try:
            engine = PaddleOCR(
                lang=options.language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            for frame in frames:
                frame.text = _paddle_texts(list(engine.predict(frame.path)))
        except TypeError:
            # PaddleOCR 2.x compatibility; 3.x uses the pipeline above.
            engine = PaddleOCR(lang=options.language, use_angle_cls=True)
            for frame in frames:
                frame.text = _paddle_texts(engine.ocr(frame.path, cls=True))
        return frames


_engines: dict[str, Callable[[], OCREngine]] = {
    RapidOCREngine.name: RapidOCREngine,
    PaddleOCREngine.name: PaddleOCREngine,
}


def register_engine(name: str, factory: Callable[[], OCREngine], *, replace: bool = False) -> None:
    if name in _engines and not replace:
        raise ValueError(f"OCR engine 已存在: {name}")
    _engines[name] = factory


def available_engines() -> tuple[str, ...]:
    return tuple(sorted(_engines))


def create_engine(name: str) -> OCREngine:
    try:
        return _engines[name]()
    except KeyError as exc:
        raise ValueError(f"未知 OCR engine: {name}；可选: {', '.join(available_engines())}") from exc


def recognize(
    frames: list[Frame], options: OCROptions, engine: str = "rapidocr"
) -> list[Frame]:
    return create_engine(engine).recognize(frames, options)
