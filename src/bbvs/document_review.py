"""Resumable chapter-local audits, structural repairs and visual reselection."""
from copy import deepcopy
import base64
import json
from pathlib import Path

from .io import read_json, write_json
from .llm import parse_json_content
from .teaching_cache import fingerprint, figure_dependencies
from .teaching_document import validate, SIGNALS

VERSION = "document-review-v2"


def review_visuals(document, client, model, directory=None):
    figures = {f["figure_id"]: f for f in document["figures"]}
    issues = []
    for chapter in document["chapters"]:
        blocks = chapter["blocks"]
        for index, block in enumerate(blocks):
            if block["kind"] != "figure":
                continue
            figure = figures[block["figure_id"]]
            context = blocks[max(0, index - 2):index + 3]
            key = fingerprint({"version": VERSION, "figure": figure_dependencies([figure]),
                               "context": context, "model": model})
            target = Path(directory) / f"visual-{key}.json" if directory else None
            if target and target.exists():
                decision = read_json(target)
            else:
                mime = "image/png" if Path(figure["path"]).suffix.lower() == ".png" else "image/jpeg"
                content = [{"type": "text", "text": "Independently verify the final image supports nearby explanation "
                    "and every referenced label is visible/readable. Return JSON accepted(boolean), reason.\n" +
                    json.dumps(context, ensure_ascii=False)}, {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(Path(figure["path"]).read_bytes()).decode()}}]
                decision = parse_json_content(client.chat(model=model, max_tokens=1024,
                    response_format={"type": "json_object"}, messages=[{"role": "user", "content": content}]))
                if not isinstance(decision.get("accepted"), bool):
                    raise ValueError("visual review must return an accepted boolean")
                if target:
                    write_json(target, decision)
            if not decision["accepted"]:
                issues.append({"block_id": block["id"], "severity": "error", "dimension": "visual",
                    "action": "reselect_figure", "reason": decision.get("reason", "unverified image")})
    return issues


def _structural_repair(local, issue, source, knowledge, client, model, progress, save):
    """Checkpoint the short plan and each long body separately."""
    chapter = local["chapters"][0]
    if "plan" not in progress:
        progress["plan"] = parse_json_content(client.chat(model=model, max_tokens=4096,
            response_format={"type": "json_object"}, messages=[{"role": "user", "content":
            "Repair the teaching structure. Return short JSON mode(insert|replace), title, blocks. "
            "Use replace only for replan_chapter; otherwise insert only missing explanations. Each block has "
            "kind(paragraph|list|callout|dialogue|figure|summary), title, purpose, evidence_ids; callout adds "
            "signal from " + "|".join(SIGNALS) + "; dialogue also source_interval; figure adds an existing figure_id. "
            "A replacement must preserve correct content and end with summary. No long text, formulas or code.\n" +
            json.dumps({"issue": issue, "chapter": chapter, "knowledge": knowledge,
                        "speech": [u.speech for u in source], "figures": local["figures"]}, ensure_ascii=False)}]))
        progress["blocks"] = []
        save()
    plan = progress["plan"]
    if plan.get("mode") not in {"insert", "replace"} or not isinstance(plan.get("blocks"), list) or not plan["blocks"]:
        raise ValueError("invalid structural repair plan")
    if plan["mode"] == "replace" and issue.get("action") != "replan_chapter":
        raise ValueError("insertion repair cannot replace a chapter")
    for index, spec in enumerate(plan["blocks"][len(progress["blocks"]):], len(progress["blocks"])):
        block = {**spec, "id": "repair-" + fingerprint({"issue": issue, "index": index,
            "chapter": [chapter.get("chapter_id"), chapter["title"], chapter.get("start"), chapter.get("end")],
            "attempt": progress.get("attempt", 0)})}
        if block["kind"] != "figure":
            block["text"] = client.chat(model=model, max_tokens=4096, messages=[{"role": "user", "content":
                "Write this repaired Chinese teaching block as plain text, grounded in the original evidence. "
                "Do not transcribe formulas/code.\n" + json.dumps({"block": spec, "issue": issue,
                "speech": [u.speech for u in source], "knowledge": knowledge}, ensure_ascii=False)}]).strip()
        progress["blocks"].append(block)
        save()
    trial = deepcopy(local)
    repaired = trial["chapters"][0]
    if plan["mode"] == "replace":
        repaired["blocks"] = deepcopy(progress["blocks"])
        repaired["title"] = plan.get("title") or repaired["title"]
    else:
        anchor = issue.get("after_block_id")
        position = next((i + 1 for i, b in enumerate(repaired["blocks"]) if b["id"] == anchor), None)
        if position is None:
            position = next((i for i, b in enumerate(repaired["blocks"]) if b["kind"] == "summary"), len(repaired["blocks"]))
        repaired["blocks"][position:position] = deepcopy(progress["blocks"])
    validate(trial)
    local["chapters"][0] = repaired


def review(document, units, client, model, rounds=2, maps=(), *, directory=None,
           figure_repair=None, visual_check=None, visual_identity=None):
    """Only changed chapters are audited; resume after each successful repair."""
    all_history, all_issues = [], []
    for chapter_index, original in enumerate(document["chapters"]):
        if "start" in original and "end" in original:
            source = [u for u in units if u.end > original["start"] and u.start < original["end"]]
        else:
            source = list(units)
        knowledge = [{"id": k.id, "content": k.content, "role": k.control.role.value,
                      "importance": k.control.importance.value} for m in maps
                     if any(u.unit_id == m.unit_id for u in source) for k in m.knowledge_items]
        figure_ids = {b.get("figure_id") for b in original["blocks"] if b["kind"] == "figure"}
        figures = [f for f in document["figures"] if f["figure_id"] in figure_ids or
                   ("start" in original and original["start"] <= f["timestamp"] < original["end"])]
        unresolved = [r for r in document.get("unresolved", []) if "start" not in original or
                      (r.get("request", r).get("start", 0) < original["end"] and
                       r.get("request", r).get("end", 0) > original["start"])]
        key = fingerprint({"version": VERSION, "chapter": original, "source": source, "knowledge": knowledge,
            "figures": figure_dependencies(figures), "unresolved": unresolved,
            "model": model, "rounds": rounds, "visual": visual_identity})
        target = Path(directory) / f"chapter-{key}.json" if directory else None
        def repaired_block_path(block_id):
            if not directory:
                return None
            index = next((i for i, b in enumerate(original["blocks"]) if b["id"] == block_id), None)
            if index is None:
                return None
            base_block = original["blocks"][index]
            adjacent = original["blocks"][max(0, index - 1):index + 2]
            related_ids = {b.get("figure_id") for b in adjacent if b["kind"] == "figure"}
            related_ids.update(base_block.get("figure_ids", []))
            related = [f for f in figures if f["figure_id"] in related_ids]
            dependency = fingerprint({"version": VERSION, "block": base_block, "source": source,
                "knowledge": knowledge, "figures": figure_dependencies(related), "model": model})
            return Path(directory) / "blocks" / f"{dependency}.json"

        cached = read_json(target) if target and target.exists() else None
        if cached and any(not Path(f["path"]).exists() for f in cached["document"]["figures"]):
            cached = None
        state = cached or {
            "document": {"evidence_ids": [k["id"] for k in knowledge] if maps else document["evidence_ids"], "figures": deepcopy(figures),
                         "chapters": [deepcopy(original)], "unresolved": unresolved},
            "round": 0, "phase": "audit", "issues": [], "history": [], "next_issue": 0}
        local = state["document"]
        if not cached:
            for block in local["chapters"][0]["blocks"]:
                block_path = repaired_block_path(block["id"])
                if block_path and block_path.exists() and block["kind"] != "figure":
                    block.update(read_json(block_path))

        def save():
            if target:
                write_json(target, state)

        while state["phase"] != "done":
            if state["phase"] == "audit":
                payload = parse_json_content(client.chat(model=model, max_tokens=4096,
                    response_format={"type": "json_object"}, messages=[{"role": "user", "content":
                    "Independently audit this chapter against original evidence, including essential content "
                    "omitted from its draft/outline. Return JSON issues array. Each issue has block_id or null, "
                    "dimension(coverage|grounding|visual|pedagogy), severity(error|warning), reason, "
                    "action(rewrite_block|insert_blocks|replan_chapter|reselect_figure). For unresolved visuals "
                    "supply request_id and reselect_figure; never invent an image. Missing blocks require "
                    "insert_blocks; broken ordering requires replan_chapter. Do not infer unseen image content.\n" +
                    json.dumps({"document": local, "speech": [u.speech for u in source], "knowledge": knowledge}, ensure_ascii=False)}]))
                issues = payload.get("issues")
                if not isinstance(issues, list) or any(not isinstance(i, dict) or i.get("severity") not in {"error", "warning"} for i in issues):
                    raise ValueError("invalid review issues")
                state["issues"] = issues
                state["phase"] = "visual"
                save()
            if state["phase"] == "visual":
                if visual_check:
                    state["issues"] += visual_check(local)
                pending = state.get("pending_visual_errors", {})
                existing = {(i.get("block_id"), i.get("request_id")) for i in state["issues"]}
                state["issues"] += [i for i in pending.values() if (i.get("block_id"), i.get("request_id")) not in existing]
                state["history"].append({"round": state["round"], "issues": deepcopy(state["issues"])})
                errors = [i for i in state["issues"] if i.get("severity") == "error"]
                state["phase"] = "repair" if errors and state["round"] < rounds else "done"
                state["next_issue"] = 0
                save()
            if state["phase"] == "repair":
                errors = [i for i in state["issues"] if i.get("severity") == "error"]
                for offset in range(state["next_issue"], len(errors)):
                    issue = errors[offset]
                    block = next((b for b in local["chapters"][0]["blocks"] if b["id"] == issue.get("block_id")), None)
                    action = issue.get("action")
                    if action == "reselect_figure" or (block and block["kind"] == "figure"):
                        repaired = figure_repair(local, issue, state["round"]) if figure_repair else False
                        issue_key = fingerprint([issue.get("block_id"), issue.get("request_id")])
                        pending = state.setdefault("pending_visual_errors", {})
                        if repaired:
                            pending.pop(issue_key, None)
                        else:
                            pending[issue_key] = issue
                    elif action in {"insert_blocks", "replan_chapter"} or block is None:
                        progress = state.setdefault("repair_progress", {})
                        progress.setdefault("attempt", state["round"])
                        _structural_repair(local, issue, source, knowledge, client, model, progress, save)
                        state.setdefault("outline_revisions", []).append({"issue": issue, "chapter": deepcopy(local["chapters"][0])})
                    else:
                        block["text"] = client.chat(model=model, max_tokens=4096, messages=[{"role": "user", "content":
                            "Revise only this teaching block to fix the evidence-backed issue. Return plain "
                            "Chinese text; do not transcribe formulas/code or invent facts.\n" + json.dumps({
                            "block": block, "issue": issue, "speech": [u.speech for u in source], "knowledge": knowledge}, ensure_ascii=False)}]).strip()
                        block_path = repaired_block_path(block["id"])
                        if block_path:
                            write_json(block_path, block)
                    state.pop("repair_progress", None)
                    state["next_issue"] = offset + 1
                    save()
                validate(local)
                state["round"] += 1
                state["phase"] = "audit"
                save()
        validate(local)
        document["chapters"][chapter_index] = deepcopy(local["chapters"][0])
        replacements = {f["figure_id"]: f for f in local["figures"]}
        document["figures"] = [deepcopy(replacements.get(f["figure_id"], f)) for f in document["figures"]]
        existing_ids = {f["figure_id"] for f in document["figures"]}
        document["figures"].extend(deepcopy(f) for f in local["figures"] if f["figure_id"] not in existing_ids)
        all_history += [{"chapter_id": original.get("chapter_id", str(chapter_index)), **row} for row in state["history"]]
        all_issues += [{"chapter_id": original.get("chapter_id", str(chapter_index)), **i} for i in state["issues"]]
        if state.get("outline_revisions"):
            document.setdefault("outline_revisions", []).extend(deepcopy(state["outline_revisions"]))
    validate(document)
    return {"history": all_history, "issues": all_issues,
            "status": "partial" if any(i.get("severity") == "error" for i in all_issues) else "complete"}
