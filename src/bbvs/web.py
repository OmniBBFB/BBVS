from __future__ import annotations

import html
import json
import mimetypes
import re
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, urlencode, urlparse

from .io import read_json
from .report import ReportOptions, _markdown_html, build_html
from .runner import run_source
from .settings import AppSettings

TaskRunner = Callable[..., Path]
_BVID = re.compile(r"BV[0-9A-Za-z]{10}")


def parse_bvid_input(value: str) -> str:
    value = value.strip()
    if _BVID.fullmatch(value):
        return value
    match = re.fullmatch(
        r"https?://(?:www\.)?bilibili\.com/video/(BV[0-9A-Za-z]{10})(?:[/?#].*)?", value,
    )
    if match:
        return match.group(1)
    raise ValueError("请输入有效的 BV 号或 Bilibili 视频 URL")


@dataclass(slots=True)
class SummaryJob:
    job_id: str
    bvid: str
    status: str = "queued"
    messages: list[str] = field(default_factory=list)
    run_dir: str = ""
    error: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "job_id": self.job_id, "bvid": self.bvid, "status": self.status,
            "messages": list(self.messages), "run_dir": self.run_dir, "error": self.error,
        }


class JobManager:
    """Own background BBVS runs and expose immutable progress snapshots."""

    def __init__(self, settings: AppSettings, runner: TaskRunner = run_source):
        self.settings = settings
        self.runner = runner
        self._jobs: dict[str, SummaryJob] = {}
        self._lock = threading.Lock()

    def submit(self, source: str) -> SummaryJob:
        bvid = parse_bvid_input(source)
        with self._lock:
            active = next((job for job in self._jobs.values() if job.bvid == bvid and job.status in {"queued", "running"}), None)
            if active:
                return active
            job = SummaryJob(uuid.uuid4().hex, bvid)
            self._jobs[job.job_id] = job
        threading.Thread(target=self._execute, args=(job.job_id,), daemon=True).start()
        return job

    def get(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.payload() if job else None

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.messages.append("任务已开始")

        def progress(message: str) -> None:
            with self._lock:
                self._jobs[job_id].messages.append(str(message))

        try:
            run_dir = self.runner(job.bvid, self.settings, progress=progress)
            with self._lock:
                job.status, job.run_dir = "completed", str(run_dir)
                job.messages.append("任务已完成")
        except Exception as exc:
            with self._lock:
                job.status, job.error = "failed", str(exc)
                job.messages.append(f"任务失败：{exc}")


@dataclass(frozen=True, slots=True)
class RunRecord:
    name: str
    bvid: str
    title: str
    uploader: str
    upload_date: str
    duration: float
    tags: tuple[str, ...]
    description: str
    path: Path
    keyframe_count: int
    reports: tuple[Path, ...]
    analyses: tuple[Path, ...]


class RunCatalog:
    """Read-only searchable view of BBVS run artifacts."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir.resolve()

    def all(self) -> list[RunRecord]:
        records = []
        if not self.runs_dir.is_dir():
            return records
        for metadata_path in sorted(self.runs_dir.glob("*/source/metadata.json")):
            try:
                metadata = read_json(metadata_path)
            except (OSError, ValueError):
                continue
            run_dir = metadata_path.parent.parent
            reports = tuple(sorted(run_dir.glob("reports/**/*.*")))
            reports = tuple(path for path in reports if path.suffix.lower() in {".html", ".pdf"})
            analyses = tuple(sorted(path for path in (run_dir / "analysis").glob("*") if path.is_dir()))
            records.append(RunRecord(
                name=run_dir.name,
                bvid=str(metadata.get("id") or run_dir.name.split("-", 1)[0]),
                title=str(metadata.get("title") or run_dir.name),
                uploader=str(metadata.get("uploader") or "未知作者"),
                upload_date=str(metadata.get("upload_date") or ""),
                duration=float(metadata.get("duration") or 0),
                tags=tuple(str(tag) for tag in metadata.get("tags") or []),
                description=str(metadata.get("description") or ""),
                path=run_dir.resolve(),
                keyframe_count=sum(1 for _ in run_dir.glob("keyframes/*/*.*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
                reports=reports, analyses=analyses,
            ))
        return sorted(records, key=lambda row: (row.upload_date, row.title), reverse=True)

    def search(self, query: str = "") -> list[RunRecord]:
        terms = query.casefold().split()
        if not terms:
            return self.all()
        return [row for row in self.all() if all(term in " ".join((
            row.bvid, row.title, row.uploader, row.description, *row.tags,
        )).casefold() for term in terms)]

    def get(self, name: str) -> RunRecord | None:
        return next((row for row in self.all() if row.name == name), None)

    def asset(self, run_name: str, relative: str) -> Path:
        record = self.get(run_name)
        if not record:
            raise FileNotFoundError(run_name)
        target = (record.path / relative).resolve()
        if not target.is_relative_to(record.path) or not target.is_file():
            raise FileNotFoundError(relative)
        return target


_STYLE = """
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--line:#29324a;--text:#eef2ff;--muted:#9ba8c7;--accent:#78d7ff;--warm:#ffca80}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#182747 0,transparent 35%),var(--bg);color:var(--text);font-family:Inter,"Noto Sans SC",system-ui,sans-serif;line-height:1.55}
a{color:var(--accent);text-decoration:none}.wrap{width:min(1180px,calc(100% - 32px));margin:auto}.top{padding:44px 0 22px}.brand{font-size:13px;letter-spacing:.18em;color:var(--accent);font-weight:800}.top h1{font-size:clamp(30px,5vw,58px);margin:8px 0}.muted{color:var(--muted)}
.search{display:flex;gap:10px;margin:18px 0}.search input{flex:1;background:#0d1426;border:1px solid var(--line);border-radius:14px;padding:16px 18px;color:var(--text);font-size:16px}.button{background:var(--accent);color:#07101d;border:0;border-radius:14px;padding:0 22px;font-weight:800;cursor:pointer}.taskbox{margin:22px 0;padding:20px;border:1px solid #315575;background:#102238;border-radius:18px}.taskbox h2{margin:0}.status{display:inline-block;padding:5px 10px;border-radius:999px;background:#233553;color:var(--warm)}.log{white-space:pre-wrap;background:#080d18;border:1px solid var(--line);border-radius:12px;padding:16px;min-height:220px;max-height:60vh;overflow:auto;font:13px/1.7 ui-monospace,monospace}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px;padding-bottom:48px}.card,.panel{background:linear-gradient(145deg,#172138,#11182a);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 14px 40px #0003}.card h2{font-size:19px;margin:10px 0}.eyebrow{display:flex;justify-content:space-between;color:var(--accent);font:700 12px ui-monospace,monospace}.meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:13px}.tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:14px}.tag{padding:4px 9px;border:1px solid var(--line);border-radius:999px;color:#c8d4f0;font-size:12px}
.back{display:inline-block;margin:25px 0}.hero{padding-bottom:22px}.hero h1{font-size:clamp(28px,4vw,48px);margin:8px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{margin-bottom:18px}.panel h2{margin-top:0}.links{display:grid;gap:8px}.doc{padding:12px;border:1px solid var(--line);border-radius:10px}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}.frame{background:#080d18;border:1px solid var(--line);border-radius:12px;overflow:hidden}.frame img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}.frame div{padding:8px 10px;color:var(--muted);font-size:12px}.markdown{max-width:900px}.markdown pre{overflow:auto;background:#080d18;padding:14px;border-radius:10px}.pager{display:flex;gap:12px;margin-top:18px}@media(max-width:760px){.cols{grid-template-columns:1fr}.search{flex-direction:column}.button{padding:14px}}
.not-found{min-height:100vh;display:grid;place-items:center;text-align:center}.not-found .panel{width:min(560px,100%);padding:44px 28px}.not-found h1{font-size:clamp(64px,14vw,120px);line-height:1;margin:12px 0;color:var(--accent)}.home-button{display:inline-block;margin-top:18px;background:var(--accent);color:#07101d;border-radius:14px;padding:13px 22px;font-weight:800}
"""


def _time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _page(title: str, body: str) -> str:
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>{body}</body></html>'


def render_index(catalog: RunCatalog, query: str = "", task_error: str = "") -> str:
    rows = catalog.search(query)
    cards = "".join(
        f'<a class="card" href="/run?name={quote(row.name)}"><div class="eyebrow"><span>{html.escape(row.bvid)}</span><span>{_time(row.duration)}</span></div>'
        f'<h2>{html.escape(row.title)}</h2><div class="meta"><span>{html.escape(row.uploader)}</span><span>{html.escape(row.upload_date)}</span><span>{row.keyframe_count} 帧</span></div>'
        f'<div class="tags">{"".join(f"<span class=\"tag\">{html.escape(tag)}</span>" for tag in row.tags[:6])}</div></a>'
        for row in rows
    ) or '<div class="panel">没有找到匹配的运行结果。</div>'
    error = f'<p style="color:#ff9b9b">{html.escape(task_error)}</p>' if task_error else ""
    body = f'<main class="wrap"><header class="top"><div class="brand">BBVS ARCHIVE</div><h1>视频理解结果库</h1><div class="muted">按 BV 号、标题、作者、标签或描述搜索已有结果</div><section class="taskbox"><h2>启动总结任务</h2><div class="muted">输入 BV 号或 Bilibili 视频 URL；已有任务将从断点继续。</div>{error}<form class="search" method="post" action="/jobs"><input name="source" required placeholder="BV1E1xxebEDs 或 https://www.bilibili.com/video/BV..."><button class="button">开始总结</button></form></section><form class="search"><input name="q" value="{html.escape(query)}" placeholder="搜索已有结果：BV1... / 软件工程 / 作者"><button class="button">搜索</button></form><div class="muted">共 {len(rows)} 条结果</div></header><section class="grid">{cards}</section></main>'
    return _page("BBVS 结果库", body)


def render_job(job_id: str) -> str:
    script = f"""<script>
const id={json.dumps(job_id)};
async function poll(){{const r=await fetch('/api/job?'+new URLSearchParams({{id}}));if(!r.ok)return;const j=await r.json();document.querySelector('#state').textContent=j.status;document.querySelector('#log').textContent=j.messages.join('\\n');const log=document.querySelector('#log');log.scrollTop=log.scrollHeight;if(j.status==='completed'||j.status==='failed'){{clearInterval(timer);document.querySelector('#done').hidden=false;}}}}
const timer=setInterval(poll,1000);poll();
</script>"""
    return _page("BBVS 任务进度", f'<main class="wrap"><a class="back" href="/">← 返回结果库</a><section class="panel"><div class="brand">SUMMARY JOB</div><h1>任务进度 <span id="state" class="status">queued</span></h1><pre id="log" class="log">等待任务启动…</pre><p id="done" hidden><a class="doc" href="/">查看结果库</a></p></section></main>{script}')


def render_not_found() -> str:
    body = (
        '<main class="wrap not-found"><section class="panel">'
        '<div class="brand">PAGE NOT FOUND</div><h1>404</h1>'
        '<h2>页面不存在</h2><p class="muted">你访问的页面可能已移动、删除，或地址输入有误。</p>'
        '<a class="home-button" href="/">回到主页</a>'
        '</section></main>'
    )
    return _page("404 · 页面不存在", body)


def render_detail(catalog: RunCatalog, record: RunRecord, frame_page: int = 1) -> str:
    per_page = 48
    frames = sorted(path for path in record.path.glob("keyframes/*/*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    page_count = max(1, (len(frames) + per_page - 1) // per_page)
    frame_page = min(max(frame_page, 1), page_count)
    shown = frames[(frame_page - 1) * per_page:frame_page * per_page]
    docs = list(record.reports)
    docs.extend(path / "report.md" for path in record.analyses if (path / "report.md").exists())
    doc_html = "".join(
        f'<a class="doc" href="/{"markdown" if path.suffix == ".md" else "asset"}?run={quote(record.name)}&path={quote(path.relative_to(record.path).as_posix())}" target="_blank">{html.escape(path.relative_to(record.path).as_posix())}</a>'
        for path in docs
    ) or '<span class="muted">暂无已生成报告</span>'
    analyses = "".join(
        f'<a class="doc" href="/generated?run={quote(record.name)}&analysis={quote(path.name)}" target="_blank">查看 {html.escape(path.name)} 分析报告</a>'
        for path in record.analyses
    ) or '<span class="muted">暂无分析产物</span>'
    gallery = "".join(
        f'<a class="frame" href="/asset?run={quote(record.name)}&path={quote(path.relative_to(record.path).as_posix())}" target="_blank"><img loading="lazy" src="/asset?run={quote(record.name)}&path={quote(path.relative_to(record.path).as_posix())}"><div>{html.escape(path.name)}</div></a>'
        for path in shown
    ) or '<span class="muted">暂无关键帧</span>'
    pager = "".join(
        f'<a class="doc" href="/run?name={quote(record.name)}&page={number}">{number}</a>'
        for number in range(1, page_count + 1)
    )
    body = f'<main class="wrap"><a class="back" href="/">← 返回结果库</a><header class="hero"><div class="brand">{html.escape(record.bvid)}</div><h1>{html.escape(record.title)}</h1><div class="meta"><span>作者：{html.escape(record.uploader)}</span><span>日期：{html.escape(record.upload_date)}</span><span>时长：{_time(record.duration)}</span></div><div class="tags">{"".join(f"<span class=\"tag\">{html.escape(tag)}</span>" for tag in record.tags)}</div></header><section class="cols"><div class="panel"><h2>结果文档</h2><div class="links">{doc_html}</div></div><div class="panel"><h2>分析版本</h2><div class="links">{analyses}</div></div></section><section class="panel"><h2>关键帧 <span class="muted">{len(frames)} 张 · 第 {frame_page}/{page_count} 页</span></h2><div class="gallery">{gallery}</div><div class="pager">{pager}</div></section></main>'
    return _page(record.title, body)


def make_handler(catalog: RunCatalog, jobs: JobManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._html(render_index(catalog, params.get("q", [""])[0], params.get("error", [""])[0]))
                elif parsed.path == "/job":
                    job_id = params.get("id", [""])[0]
                    if jobs.get(job_id) is None: raise FileNotFoundError
                    self._html(render_job(job_id))
                elif parsed.path == "/api/job":
                    payload = jobs.get(params.get("id", [""])[0])
                    if payload is None: raise FileNotFoundError
                    self._json(payload)
                elif parsed.path == "/run":
                    record = catalog.get(params.get("name", [""])[0])
                    if not record: raise FileNotFoundError
                    self._html(render_detail(catalog, record, int(params.get("page", ["1"])[0])))
                elif parsed.path == "/asset":
                    self._file(catalog.asset(params.get("run", [""])[0], params.get("path", [""])[0]))
                elif parsed.path == "/markdown":
                    path = catalog.asset(params.get("run", [""])[0], params.get("path", [""])[0])
                    self._html(_page(path.stem, f'<main class="wrap markdown"><a class="back" href="javascript:history.back()">← 返回</a>{_markdown_html(path.read_text(encoding="utf-8"))}</main>'))
                elif parsed.path == "/generated":
                    record = catalog.get(params.get("run", [""])[0])
                    if not record: raise FileNotFoundError
                    analysis_name = params.get("analysis", [""])[0]
                    analysis = next((path for path in record.analyses if path.name == analysis_name), None)
                    if analysis is None: raise FileNotFoundError
                    self._html(build_html(record.path, ReportOptions(max_images=24), analysis_dir=analysis))
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, ValueError, OSError):
                self._not_found()

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/jobs":
                self._not_found(); return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 4096)
                params = parse_qs(self.rfile.read(length).decode("utf-8"))
                job = jobs.submit(params.get("source", [""])[0])
            except (ValueError, UnicodeDecodeError) as exc:
                self.send_response(303); self.send_header("Location", "/?" + urlencode({"error": str(exc)})); self.end_headers(); return
            self.send_response(303); self.send_header("Location", "/job?" + urlencode({"id": job.job_id})); self.end_headers()

        def _html(self, document: str, status: int = 200) -> None:
            payload = document.encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def _not_found(self) -> None:
            self._html(render_not_found(), status=404)

        def _file(self, path: Path) -> None:
            payload = path.read_bytes()
            self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def _json(self, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(settings: AppSettings, host: str = "127.0.0.1", port: int = 8765) -> None:
    catalog = RunCatalog(settings.runs_dir)
    server = ThreadingHTTPServer((host, port), make_handler(catalog, JobManager(settings)))
    print(f"BBVS 结果库：http://{host}:{port} （数据目录：{catalog.runs_dir}）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
