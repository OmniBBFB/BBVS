from pathlib import Path

import pytest

from bbvs.archive import archive_run, run_name, safe_name
from bbvs.io import read_json, write_json


def test_run_name_keeps_id_and_sanitizes_title() -> None:
    assert run_name({"id": "BV123_p4", "title": "课程: A/B?"}) == "BV123_p4-课程 A B"
    assert safe_name("  a\n  b. ") == "a b"


def test_archive_run_renames_and_rewrites_absolute_json_paths(tmp_path: Path) -> None:
    old = tmp_path / "BV123_p4"
    frame = old / "keyframes" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpg")
    write_json(old / "source" / "metadata.json", {"id": "BV123_p4", "title": "课程/测试"})
    write_json(old / "ocr.json", [{"path": str(frame.resolve())}])

    new = archive_run(old)

    assert new.name == "BV123_p4-课程 测试"
    assert not old.exists()
    rewritten = Path(read_json(new / "ocr.json")[0]["path"])
    assert rewritten == new / "keyframes" / "frame.jpg"
    assert rewritten.exists()


def test_archive_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    old = tmp_path / "old"
    write_json(old / "source" / "metadata.json", {"id": "BV1", "title": "Title"})
    (tmp_path / "BV1-Title").mkdir()
    with pytest.raises(FileExistsError):
        archive_run(old)
