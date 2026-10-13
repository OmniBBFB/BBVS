"""Evidence-linked teaching documents; prose stays outside model JSON responses."""
from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from .io import read_json, write_json, write_text
from .llm import parse_json_content
from .teaching_cache import fingerprint, figure_dependencies, chapter_identity

VERSION = "teaching-document-v2"
KINDS = {"paragraph", "list", "callout", "dialogue", "figure", "summary"}
SIGNALS = {
    "central_claim": "important", "key_definition": "important",
    "mechanism_takeaway": "important", "background": "knowledge",
    "prerequisite": "knowledge", "intuition": "knowledge", "tradeoff": "knowledge",
    "misconception": "warning", "failure_mode": "warning", "hidden_assumption": "warning",
    "high_value_dialogue": "dialogue",
}


def validate(document):
    ids = set()
    figures = {f["figure_id"]: f for f in document["figures"]}
    if len(figures) != len(document["figures"]):
        raise ValueError("duplicate figure id")
    for figure in figures.values():
        if not Path(figure["path"]).is_file() or not 0 <= figure["start"] < figure["end"]:
            raise ValueError("missing figure asset or invalid source interval")
    evidence = set(document["evidence_ids"])
    for chapter in document["chapters"]:
        for block in chapter["blocks"]:
            if block["id"] in ids or block["kind"] not in KINDS:
                raise ValueError("duplicate block id or unknown block kind")
            ids.add(block["id"])
            if not set(block.get("evidence_ids", [])).issubset(evidence):
                raise ValueError("unknown evidence reference")
            if block["kind"] == "figure" and block.get("figure_id") not in figures:
                raise ValueError("unknown figure reference")
            if block["kind"] in {"callout", "dialogue"}:
                if block.get("signal") not in SIGNALS or not block.get("evidence_ids"):
                    raise ValueError("callout requires a pedagogical signal and evidence")
            if block["kind"] == "dialogue" and not block.get("source_interval"):
                raise ValueError("dialogue requires a source interval")
        if not any(b["kind"] == "summary" for b in chapter["blocks"]):
            raise ValueError("chapter requires a summary")


def compose(chapters, units, maps, figures, client, model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = {"version": VERSION, "figures": figures, "chapters": [],
                "evidence_ids": [k.id for m in maps for k in m.knowledge_items]}
    for index, chapter in enumerate(chapters):
        source = [u for u in units if u.end > chapter.start and u.start < chapter.end]
        available = [f for f in figures if chapter.start <= f["timestamp"] < chapter.end]
        evidence = [{"id": k.id, "content": k.content, "role": k.control.role.value,
                     "importance": k.control.importance.value} for m in maps
                    if any(u.unit_id == m.unit_id for u in source) for k in m.knowledge_items]
        chapter_id = chapter_identity(chapter)
        local = directory / chapter_id
        local.mkdir(exist_ok=True)
        chapter_source = {**asdict(chapter), "summary": chapter.summary.strip()}
        plan_key = fingerprint({"version": VERSION, "model": model, "chapter": chapter_source,
            "evidence": evidence, "source": source,
            "figures": [{"figure_id": f["figure_id"], "request_id": f.get("request_id")} for f in available]})
        plan_path = local / f"plan-{plan_key}.json"
        if plan_path.exists():
            payload = read_json(plan_path)
        else:
            payload = parse_json_content(client.chat(model=model, max_tokens=4096,
            response_format={"type": "json_object"}, messages=[{"role": "user", "content":
            "Plan a Chinese teaching chapter as short JSON metadata only: blocks array. Each block has "
            "kind(paragraph|list|callout|dialogue|figure|summary), title, purpose, evidence_ids; "
            "callout adds signal(central_claim|key_definition|mechanism_takeaway|background|prerequisite|"
            "intuition|tradeoff|misconception|failure_mode|hidden_assumption). dialogue adds "
            "signal=high_value_dialogue and source_interval. figure adds figure_id. "
            "Place figures next to the exact explanation they support. Include motivation, mechanism, examples "
            "when applicable and a final summary. Avoid redundant boxes. Do not transcribe formulas/code.\n"
            + json.dumps({"chapter": chapter.summary, "evidence": evidence, "figures": available}, ensure_ascii=False)}]))
            write_json(plan_path, payload)
        row = {"chapter_id": chapter_id, "title": chapter.title, "start": chapter.start, "end": chapter.end, "blocks": []}
        for number, block in enumerate(payload["blocks"]):
            block = {**block, "id": f"c{chapter_id}b{number:03d}"}
            if not set(block.get("evidence_ids", [])).issubset({e["id"] for e in evidence}):
                raise ValueError("block cites evidence outside its chapter")
            if block["kind"] == "figure" and block.get("figure_id") not in {f["figure_id"] for f in available}:
                raise ValueError("block cites a figure outside its chapter")
            if block["kind"] != "figure":
                # Only prose next to a figure or citing it depends on that image.
                neighbours = payload["blocks"][max(0, number - 1):number + 2]
                ids = {b.get("figure_id") for b in neighbours if b["kind"] == "figure"}
                ids.update(block.get("figure_ids", []))
                related = [f for f in available if f["figure_id"] in ids]
                body_key = fingerprint({"version": VERSION, "model": model, "plan": block,
                    "source": source, "evidence": evidence, "draft": chapter.summary.strip(),
                    "figures": figure_dependencies(related)})
                body_path = local / f"{block['id']}-{body_key}.txt"
                if body_path.exists():
                    body = body_path.read_text(encoding="utf-8")
                else:
                    body = client.chat(model=model, max_tokens=4096, messages=[{"role": "user", "content":
                    "Write this Chinese teaching block as plain text from the supplied evidence. Explain clearly "
                    "without adding facts. For dialogue preserve only reliable short original turns and speakers. "
                    "Do not invent formulas/code.\n" + json.dumps({"plan": block, "draft": chapter.summary,
                    "speech": [u.speech for u in source], "evidence": evidence, "figures": related}, ensure_ascii=False)}]).strip()
                    write_text(body_path, body)
                block["text"] = body
            row["blocks"].append(block)
        trial = {**document, "chapters": [row]}
        validate(trial)
        write_json(local / "chapter.json", row)
        document["chapters"].append(row)
    validate(document)
    return document


def render_html(document):
    from .report import _image_uri
    validate(document)
    figures = {f["figure_id"]: f for f in document["figures"]}
    parts = []
    if document.get("cover_path") and Path(document["cover_path"]).is_file():
        parts.append('<img style="width:100%" src="' + _image_uri(Path(document["cover_path"])) + '">')
    for chapter in document["chapters"]:
        parts.append(f"<section><h2>{html.escape(chapter['title'])}</h2>")
        for b in chapter["blocks"]:
            if b["kind"] == "figure":
                f = figures[b["figure_id"]]
                parts.append(f'<figure><img style="width:100%" src="{_image_uri(Path(f["path"]))}">'
                             f'<figcaption>{html.escape(f["caption"])} · {f["start"]:.1f}–{f["end"]:.1f}s</figcaption></figure>')
            else:
                style = SIGNALS.get(b.get("signal"), "")
                colors = {"important": "#fff4c4", "knowledge": "#eaf3ff", "warning": "#ffe9e9", "dialogue": "#eaf8ed"}
                parts.append(f'<div style="background:{colors.get(style, "transparent")};padding:8px">'
                             f'<h3>{html.escape(b.get("title", ""))}</h3><p>'
                             + html.escape(b.get("text", "")).replace("\n", "<br>") + '</p></div>')
        parts.append("</section>")
    if document.get("synthesis"):
        parts.append("<h2>总结与延伸</h2><p>" + html.escape(document["synthesis"]).replace("\n", "<br>") + "</p>")
    return "".join(parts)
