from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _payload(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不支持序列化的配置类型: {type(value).__name__}")


def _cache_config(value: Any) -> Any:
    """Remove credentials: they authorize a run but do not define its result."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: _cache_config(item) for key, item in value.items() if key not in {"api_key", "token"}}
    if isinstance(value, (list, tuple)):
        return [_cache_config(item) for item in value]
    return value


def variant(label: str, config: Any) -> str:
    """Return a readable, stable directory name for an exact configuration."""
    encoded = json.dumps(_cache_config(config), default=_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]
    readable = re.sub(r"[^0-9A-Za-z._-]+", "-", label).strip("-._").lower() or "default"
    return f"{readable[:80]}--{digest}"


def step_dir(run_dir: Path, step: str, label: str, config: Any) -> Path:
    return run_dir / step / variant(label, config)


def find_inputs(run_dir: Path, step: str, filename: str, legacy_pattern: str) -> list[Path]:
    """Discover structured artifacts, while retaining read compatibility with old runs."""
    current = sorted((run_dir / step).glob(f"*/{filename}"))
    return current or sorted(run_dir.glob(legacy_pattern))
