from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import DependencyError
from .io import read_json

REPORT_VERSION = "vlm-verified-visuals-v2"


@dataclass(frozen=True, slots=True)
class ReportOptions:
    include_transcript: bool = False
    max_images: int | None = None
    expect_vision: bool = True


class PdfRenderer(Protocol):
    def render(self, document: str, output: Path, base_url: Path) -> None: ...


class WeasyPrintRenderer:
    def render(self, document: str, output: Path, base_url: Path) -> None:
        try:
            from weasyprint import HTML
        except ImportError as exc:
            raise DependencyError("缺少 PDF 引擎，请运行: uv sync --extra report") from exc
        HTML(string=document, base_url=str(base_url)).write_pdf(str(output))


def _load(path: Path, default: Any) -> Any:
    return read_json(path) if path.exists() else default


def _time(seconds: float | int | None) -> str:
    value = max(0, int(seconds or 0))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _text(value: Any) -> str:
    return html.escape(str(value or ""))


def _timestamp_link(url: str, seconds: float) -> str:
    separator = "&" if "?" in url else "?"
    target = html.escape(f"{url}{separator}t={int(seconds)}") if url else "#"
    return f'<a href="{target}">{_time(seconds)}</a>'


def _image_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _representative_frames(run_dir: Path, limit: int | None, frames_path: Path | None = None) -> list[dict[str, Any]]:
    candidates = [frames_path] if frames_path else sorted(run_dir.glob("ocr-*.json"))
    if not candidates or limit == 0:
        return []
    frames = [row for row in read_json(candidates[0]) if Path(row["path"]).exists()]
    informative = [row for row in frames if len(row.get("text", [])) >= 8] or frames
    if limit is None or len(informative) <= limit:
        return informative
    indices = [round(index * (len(informative) - 1) / (limit - 1)) for index in range(limit)] if limit > 1 else [len(informative) // 2]
    return [informative[index] for index in indices]


def _visual_timeline_cards(
    run_dir: Path, analysis: Path, limit: int | None, frames_path: Path | None = None,
) -> list[dict[str, Any]]:
    visual_path = analysis / "visual-analysis.json"
    frame_candidates = [frames_path] if frames_path else sorted(run_dir.glob("ocr-*.json"))
    if limit == 0 or not visual_path.exists() or not frame_candidates:
        return []
    frames = [row for row in read_json(frame_candidates[0]) if Path(row["path"]).exists()]
    cards = []
    for visual in read_json(visual_path):
        if not frames:
            break
        frame = min(frames, key=lambda row: abs(float(row["timestamp"]) - float(visual["timestamp"])))
        if abs(float(frame["timestamp"]) - float(visual["timestamp"])) <= 1.0:
            cards.append({**visual, "path": frame["path"], "ocr_text": frame.get("text", [])})
    if limit is None or len(cards) <= limit:
        return cards
    indices = [round(index * (len(cards) - 1) / (limit - 1)) for index in range(limit)] if limit > 1 else [len(cards) // 2]
    return [cards[index] for index in indices]


def _chapter_html(row: dict[str, Any], url: str) -> str:
    title_zh = row.get("title_zh")
    summary_zh = row.get("summary_zh")
    points_zh = row.get("key_points_zh", [])
    parts = [f'<section class="chapter"><h3>{_text(row.get("title"))}</h3>']
    if title_zh:
        parts.append(f'<div class="translation"><div class="translation-label">中文标题</div>{_text(title_zh)}</div>')
    parts.append(
        f'<div class="time">{_timestamp_link(url, row.get("start", 0))}–'
        f'{_timestamp_link(url, row.get("end", 0))}</div><p>{_text(row.get("summary"))}</p>'
    )
    if summary_zh:
        parts.append(f'<div class="translation"><div class="translation-label">中文翻译</div>{_text(summary_zh)}</div>')
    parts.append('<ul>' + "".join(f'<li>{_text(point)}</li>' for point in row.get("key_points", [])) + '</ul>')
    if points_zh:
        parts.append(
            '<div class="translation"><div class="translation-label">要点翻译</div><ul>'
            + "".join(f'<li>{_text(point)}</li>' for point in points_zh) + '</ul></div>'
        )
    parts.append('</section>')
    return "".join(parts)


_STYLE = """
@page { size: A4; margin: 18mm 16mm 20mm; @bottom-center { content: counter(page); color: #64748b; font-size: 9pt; } }
* { box-sizing: border-box; }
body { font-family: "Noto Sans CJK SC", "Noto Sans CJK", sans-serif; color: #18212f; font-size: 10.5pt; line-height: 1.62; margin: 0; }
h1, h2, h3 { color: #102a43; page-break-after: avoid; }
h1 { font-size: 27pt; line-height: 1.2; margin: 0 0 10mm; }
h2 { border-bottom: 2px solid #2563eb; padding-bottom: 2mm; margin-top: 10mm; }
h3 { margin-bottom: 2mm; }
.cover { min-height: 235mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.eyebrow { color: #2563eb; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.meta { color: #52606d; font-size: 11pt; }
.badge { display: inline-block; background: #e8f0fe; color: #174ea6; border-radius: 99px; padding: 1mm 3mm; margin: 1mm; }
.warning { background: #fff7d6; border-left: 4px solid #e5a100; padding: 3mm 4mm; margin: 5mm 0; }
.chapter { break-inside: auto; margin: 6mm 0; padding: 4mm; border: 1px solid #d9e2ec; border-radius: 3mm; }
.translation { color: #334e68; background: #f6f9fc; border-left: 3px solid #93b4e8; padding: 2mm 3mm; }
.translation-label { color: #2563eb; font-size: 8pt; font-weight: 700; letter-spacing: .06em; }
.time { color: #2563eb; font-weight: 700; }
.visual-timeline { border-left: 2px solid #93b4e8; padding-left: 5mm; }
.visual-card { position: relative; break-inside: avoid; display: grid; grid-template-columns: 44% 56%; gap: 5mm; margin: 6mm 0; padding: 4mm; border: 1px solid #d9e2ec; border-radius: 3mm; }
.visual-card img { width: 100%; height: auto; display: block; }
.visual-copy p { margin: 1.5mm 0 3mm; }
.caption { font-size: 8.5pt; color: #52606d; margin-top: 1mm; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; }
th, td { border: 1px solid #d9e2ec; padding: 2mm; vertical-align: top; }
th { background: #f0f4f8; text-align: left; }
a { color: #2563eb; text-decoration: none; }
.transcript { font-size: 8.7pt; }
.transcript p { margin: 1.5mm 0; }
"""


def build_html(
    run_dir: Path, options: ReportOptions = ReportOptions(), *,
    analysis_dir: Path | None = None, transcript_path: Path | None = None,
    frames_path: Path | None = None,
) -> str:
    source = run_dir / "source"
    analysis = analysis_dir or run_dir / "analysis"
    metadata = _load(source / "metadata.json", {})
    summary = _load(analysis / "summary.json", {})
    chapters = _load(analysis / "chapters.json", [])
    terms = _load(analysis / "terminology.json", [])
    corrections = _load(analysis / "corrections.json", [])
    timeline = _load(analysis / "timeline.json", [])
    expected_stages = [
        ("全局摘要", bool(summary)), ("章节摘要", bool(chapters)),
        ("转录校正", (analysis / "verified-transcript.json").exists()),
    ]
    if options.expect_vision:
        expected_stages.append(("视觉分析", (analysis / "visual-analysis.json").exists()))
    missing = [name for name, present in expected_stages if not present]
    url = str(metadata.get("webpage_url", ""))
    duration = metadata.get("duration") or (timeline[-1]["end"] if timeline else 0)
    tags = "".join(f'<span class="badge">{_text(tag)}</span>' for tag in metadata.get("tags", []))
    warning = f'<div class="warning">本报告基于已有产物生成；尚未运行：{_text("、".join(missing))}。</div>' if missing else ""

    summary_html = ""
    if summary:
        summary_html = f'<h2>内容摘要</h2><p>{_text(summary.get("summary"))}</p>'
        if summary.get("summary_zh"):
            summary_html += f'<div class="translation"><div class="translation-label">中文翻译</div>{_text(summary.get("summary_zh"))}</div>'
        for key, title in (("key_concepts", "关键概念"), ("takeaways", "核心结论")):
            values = summary.get(key, [])
            if values:
                summary_html += f'<h3>{title}</h3><ul>' + "".join(f'<li>{_text(value)}</li>' for value in values) + "</ul>"
            translated = summary.get(f"{key}_zh", [])
            if translated:
                summary_html += '<div class="translation"><div class="translation-label">中文翻译</div><ul>' + "".join(f'<li>{_text(value)}</li>' for value in translated) + "</ul></div>"

    chapters_html = ""
    if chapters:
        chapters_html = "<h2>章节</h2>" + "".join(_chapter_html(row, url) for row in chapters)

    term_rows = "".join(
        f'<tr><td>{_text(row.get("canonical_name"))}</td><td>{_text(row.get("category"))}</td>'
        f'<td>{float(row.get("confidence", 0)):.2f}</td><td>{_text("; ".join(e.get("text", "") for e in row.get("evidence", [])))}</td></tr>'
        for row in terms
    )
    terms_html = f'<h2>术语表</h2><table><thead><tr><th>术语</th><th>类别</th><th>置信度</th><th>证据</th></tr></thead><tbody>{term_rows}</tbody></table>' if terms else ""

    frames = (
        _visual_timeline_cards(run_dir, analysis, options.max_images, frames_path)
        if options.expect_vision else []
    )
    gallery = ""
    if frames:
        figures = "".join(
            f'<section class="visual-card"><div><img src="{_image_uri(Path(row["path"]))}"><div class="caption">'
            f'{_timestamp_link(url, row.get("timestamp", 0))} · {_text(row.get("visual_type", ""))}</div></div>'
            f'<div class="visual-copy"><div class="time">{_timestamp_link(url, row.get("timestamp", 0))}</div>'
            f'<p>{_text(row.get("summary"))}</p>'
            + (f'<div class="translation"><div class="translation-label">中文翻译</div>{_text(row.get("summary_zh"))}</div>' if row.get("summary_zh") else '')
            + '</div></section>'
            for row in frames
        )
        gallery = f'<h2>图文时间轴</h2><div class="visual-timeline">{figures}</div>'

    corrections_html = ""
    if corrections:
        rows = "".join(
            f'<tr><td>{_timestamp_link(url, row.get("start", 0))}</td><td>{_text(row.get("original"))}</td><td>{_text(row.get("corrected"))}</td><td>{float(row.get("confidence", 0)):.2f}</td></tr>'
            for row in corrections
        )
        corrections_html = f'<h2>转录校正</h2><table><tr><th>时间</th><th>原文</th><th>校正</th><th>置信度</th></tr>{rows}</table>'

    transcript_html = ""
    if options.include_transcript:
        candidates = [analysis / "verified-transcript.json"]
        candidates += [transcript_path] if transcript_path else sorted(run_dir.glob("asr-*.json"))
        transcript = next((_load(path, {}) for path in candidates if path.exists()), {})
        rows = "".join(
            f'<p><span class="time">{_timestamp_link(url, row.get("start", 0))}</span> {_text(row.get("text"))}</p>'
            for row in transcript.get("segments", [])
        )
        transcript_html = f'<section class="transcript"><h2>完整转录</h2>{rows}</section>' if rows else ""

    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{_text(metadata.get("title", "视频报告"))}</title><style>{_STYLE}</style></head><body>
<section class="cover"><div class="eyebrow">Video Understanding Report</div><h1>{_text(metadata.get("title", "视频分析报告"))}</h1>
<div class="meta">时长 {_time(duration)} · 上传者 {_text(metadata.get("uploader", "未知"))} · 日期 {_text(metadata.get("upload_date", ""))}</div><div>{tags}</div>{warning}</section>
{summary_html}{chapters_html}{terms_html}{gallery}{corrections_html}{transcript_html}
</body></html>'''


def export_report(
    run_dir: Path, output: Path, options: ReportOptions = ReportOptions(),
    renderer: PdfRenderer | None = None, *, analysis_dir: Path | None = None,
    transcript_path: Path | None = None, frames_path: Path | None = None,
) -> Path:
    document = build_html(
        run_dir, options, analysis_dir=analysis_dir,
        transcript_path=transcript_path, frames_path=frames_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".html":
        output.write_text(document, encoding="utf-8")
    elif output.suffix.lower() == ".pdf":
        (renderer or WeasyPrintRenderer()).render(document, output, run_dir)
    else:
        raise ValueError("报告输出必须使用 .html 或 .pdf 扩展名")
    return output
