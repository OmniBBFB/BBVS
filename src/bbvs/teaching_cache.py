"""Content fingerprints for local document dependencies, never renderer settings."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    def normalize(item):
        if is_dataclass(item):
            return normalize(asdict(item))
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item
    return hashlib.sha256(json.dumps(normalize(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()[:24]


def figure_dependencies(figures):
    return [{**f, "asset_digest": hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest()}
            for f in figures]


def chapter_identity(chapter):
    return fingerprint({"title": chapter.title, "start": chapter.start, "end": chapter.end})
