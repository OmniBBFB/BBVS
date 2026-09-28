from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest

from bbvs.io import write_json
from bbvs.config import Endpoint, Services
from bbvs.settings import AppSettings
from bbvs.web import (
    JobManager, RunCatalog, parse_bvid_input, render_detail, render_index, render_not_found,
)


def make_run(root: Path, bvid: str, title: str, uploader: str) -> Path:
    run = root / f"{bvid}-{title}"
    write_json(run / "source/metadata.json", {
        "id": bvid, "title": title, "uploader": uploader,
        "upload_date": "20260921", "duration": 125, "tags": ["软件工程"],
        "description": "版本管理课程",
    })
    frame = run / "keyframes/variant/frame_001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpg")
    (run / "analysis/v1").mkdir(parents=True)
    (run / "analysis/v1/report.md").write_text("# 总结\n", encoding="utf-8")
    return run


def test_catalog_searches_bvid_title_uploader_tags_and_description(tmp_path: Path) -> None:
    make_run(tmp_path, "BV1E1xxebEDs", "软件仓库管理", "绿导师")
    catalog = RunCatalog(tmp_path)

    for query in ("BV1E1xxebEDs", "仓库", "绿导师", "软件工程", "版本管理"):
        assert [row.bvid for row in catalog.search(query)] == ["BV1E1xxebEDs"]
    assert catalog.search("不存在") == []


def test_catalog_restricts_assets_to_the_selected_run(tmp_path: Path) -> None:
    run = make_run(tmp_path, "BV1E1xxebEDs", "课程", "作者")
    catalog = RunCatalog(tmp_path)

    assert catalog.asset(run.name, "keyframes/variant/frame_001.jpg").is_file()
    with pytest.raises(FileNotFoundError):
        catalog.asset(run.name, "../../secret.txt")


def test_pages_render_search_metadata_documents_and_frames(tmp_path: Path) -> None:
    run = make_run(tmp_path, "BV1E1xxebEDs", "课程<script>", "作者")
    catalog = RunCatalog(tmp_path)
    record = catalog.get(run.name)
    assert record is not None

    index = render_index(catalog, "作者")
    detail = render_detail(catalog, record)

    assert "BV1E1xxebEDs" in index and "作者" in index
    assert "课程&lt;script&gt;" in index
    assert "report.md" in detail and "frame_001.jpg" in detail
    assert "课程<script>" not in detail


def test_job_manager_accepts_url_reports_progress_and_deduplicates(tmp_path: Path) -> None:
    release = Event()

    def runner(bvid, settings, progress):
        progress("[1/7] 下载")
        release.wait(1)
        return settings.runs_dir / f"{bvid}-课程"

    settings = AppSettings(Services(Endpoint("http://llm/v1", "m")), runs_dir=tmp_path)
    jobs = JobManager(settings, runner=runner)
    first = jobs.submit("https://www.bilibili.com/video/BV1E1xxebEDs?p=1")
    second = jobs.submit("BV1E1xxebEDs")
    assert first.job_id == second.job_id
    release.set()

    deadline = monotonic() + 1
    while jobs.get(first.job_id)["status"] != "completed" and monotonic() < deadline:  # type: ignore[index]
        sleep(0.01)
    payload = jobs.get(first.job_id)
    assert payload is not None
    assert payload["status"] == "completed"
    assert "[1/7] 下载" in payload["messages"]
    assert str(tmp_path) in str(payload["run_dir"])


def test_task_input_rejects_non_bilibili_urls() -> None:
    assert parse_bvid_input("BV1E1xxebEDs") == "BV1E1xxebEDs"
    with pytest.raises(ValueError, match="Bilibili"):
        parse_bvid_input("https://example.com/BV1E1xxebEDs")


def test_not_found_page_has_home_link() -> None:
    page = render_not_found()

    assert "404" in page
    assert "页面不存在" in page
    assert '<a class="home-button" href="/">回到主页</a>' in page
