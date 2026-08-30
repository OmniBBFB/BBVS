from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import read_json, write_json

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_SPACE = re.compile(r"\s+")


def safe_name(value: str, max_length: int = 100) -> str:
    cleaned = _SPACE.sub(" ", _UNSAFE.sub(" ", value)).strip(" .-")
    return cleaned[:max_length].rstrip(" .-") or "untitled"


def run_name(metadata: dict[str, Any]) -> str:
    video_id = safe_name(str(metadata.get("id") or "video"), 40)
    title = safe_name(str(metadata.get("title") or "untitled"), 100)
    return f"{video_id}-{title}"


def _replace_paths(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_paths(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, old, new) for key, item in value.items()}
    return value


def archive_run(run_dir: Path, runs_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "source" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"缺少运行元数据: {metadata_path}")
    destination = (runs_dir.resolve() if runs_dir else run_dir.parent) / run_name(read_json(metadata_path))
    if destination == run_dir:
        return run_dir
    if destination.exists():
        raise FileExistsError(f"目标运行目录已存在: {destination}")
    old = str(run_dir)
    run_dir.rename(destination)
    new = str(destination)
    for path in destination.rglob("*.json"):
        payload = read_json(path)
        updated = _replace_paths(payload, old, new)
        if updated != payload:
            write_json(path, updated)
    return destination
