from __future__ import annotations

import re
from pathlib import Path

from .io import read_json, write_json, write_text
from .models import Chapter


def markdown_title(content: str, fallback: str) -> str:
    match = re.search(r"^#{1,6}\s+(.+?)\s*$", content, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def text_title(content: str, fallback: str) -> str:
    match = re.search(r"^\s*(?:标题[：:]\s*)?(.+?)\s*$", content, re.MULTILINE)
    return re.sub(r"^#{1,6}\s+", "", match.group(1)).strip() if match else fallback


def write_text_chapters(analysis_dir: Path, chapters: list[Chapter]) -> None:
    chapter_dir = analysis_dir / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, chapter in enumerate(chapters, 1):
        relative = Path("chapters") / f"{index:03d}.txt"
        write_text(analysis_dir / relative, chapter.summary.strip() + "\n")
        manifest.append({
            "file": relative.as_posix(), "title": chapter.title,
            "start": chapter.start, "end": chapter.end,
        })
    write_json(analysis_dir / "chapters-manifest.json", manifest)


def write_markdown_chapters(analysis_dir: Path, chapters: list[Chapter]) -> None:
    chapter_dir = analysis_dir / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, chapter in enumerate(chapters, 1):
        relative = Path("chapters") / f"{index:03d}.md"
        write_text(analysis_dir / relative, chapter.summary.strip() + "\n")
        manifest.append({
            "file": relative.as_posix(), "title": chapter.title,
            "start": chapter.start, "end": chapter.end,
        })
    write_json(analysis_dir / "chapters-manifest.json", manifest)


def load_markdown_chapters(analysis_dir: Path) -> list[Chapter]:
    manifest_path = analysis_dir / "chapters-manifest.json"
    if not manifest_path.exists():
        legacy_path = analysis_dir / "chapters.json"
        return [Chapter(**row) for row in read_json(legacy_path)] if legacy_path.exists() else []
    chapters = []
    for row in read_json(manifest_path):
        path = analysis_dir / str(row["file"])
        if not path.is_file():
            break
        chapters.append(Chapter(
            title=str(row.get("title", "")), start=float(row.get("start", 0)),
            end=float(row.get("end", 0)), summary=path.read_text(encoding="utf-8"),
        ))
    return chapters


def load_text_chapters(analysis_dir: Path) -> list[Chapter]:
    """Load current plain-text chapters and legacy Markdown/JSON manifests."""
    return load_markdown_chapters(analysis_dir)
