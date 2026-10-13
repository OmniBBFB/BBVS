"""Resolve semantic visual requests by sampling the original media."""
import base64
import hashlib
import json
from pathlib import Path

from .io import read_json, write_json
from .llm import parse_json_content
from .process import require_executable, run
from .media import probe

VERSION = "visual-resolution-v2"


def collect_requests(units, maps, outline):
    """Merge map and editor requests, retaining only selected topic evidence."""
    unit_by_id = {u.unit_id: u for u in units}
    selected = {item for chapter in outline for topic in chapter.topics
                for item in topic.required_items + topic.supporting_items}
    requests, seen = [], set()
    for content in maps:
        if selected and not any(k.id in selected for k in content.knowledge_items):
            continue
        unit = unit_by_id[content.unit_id]
        source = content.visual_requests
        if content.formulas_or_code and not source:
            source = [{"description": "原始公式或代码课件截图", "start": unit.start,
                       "end": unit.end, "importance": "required"}]
        for row in source:
            requests.append({**row, "start": row.get("start", unit.start), "end": row.get("end", unit.end)})
    for chapter in outline:
        start, end = unit_by_id[chapter.start_unit].start, unit_by_id[chapter.end_unit].end
        requests.extend({**row, "start": row.get("start", start), "end": row.get("end", end)}
                        for row in chapter.visual_requests)
    result = []
    for row in requests:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append({**row, "request_id": hashlib.sha256(key.encode()).hexdigest()[:16]})
    return result


def resolve(requests, video, directory, client, model, interval=3.0, attempts=2, max_candidates=120):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    resolved, unresolved = [], []
    media = Path(video).stat()
    duration = probe(Path(video)).duration
    for request in requests:
        if not 0 <= float(request["start"]) < float(request["end"]):
            raise ValueError("visual request requires a valid time interval")
        key = hashlib.sha256(json.dumps({"request": request, "model": model, "version": VERSION,
            "media": [str(video), media.st_size, media.st_mtime_ns], "interval": interval,
            "attempts": attempts, "max_candidates": max_candidates}, sort_keys=True).encode()).hexdigest()[:20]
        cache = directory / f"{key}.json"
        cached = read_json(cache) if cache.exists() else None
        if cached and cached.get("status") == "resolved" and Path(cached["path"]).exists():
            result = cached
        else:
            result = {"request": request, "status": "unresolved", "reason": "no acceptable frame"}
            for attempt in range(attempts):
                start = max(0, float(request["start"]) - 8 * (attempt + 1))
                end = float(request["end"]) + 8 * (attempt + 1)
                if duration is not None:
                    end = min(end, duration)
                candidates = []
                step = interval / (attempt + 1)
                count = int((end - start) / step) + 1
                if count > max_candidates:
                    count = max_candidates
                    step = (end - start) / max(1, count - 1)
                for number in range(count):
                    timestamp = start + number * step
                    if any(abs(timestamp - rejected) <= step / 2 for rejected in request.get("rejected_timestamps", [])):
                        continue
                    path = directory / f"{key}-{attempt}-{number}.jpg"
                    run([require_executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                         "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", "-q:v", "2", str(path)])
                    if path.exists():
                        candidates.append({"id": len(candidates), "path": str(path), "timestamp": timestamp})
                winners = []
                for offset in range(0, len(candidates), 8):
                    group = candidates[offset:offset + 8]
                    content = [{"type": "text", "text":
                        "Compare candidates against the semantic request. Return JSON with candidate_id "
                        "or null, reason, caption. Select only relevant, fully revealed, readable frames "
                        "with every required_content element visible.\n" + json.dumps(request, ensure_ascii=False)}]
                    for c in group:
                        content.extend([{"type": "text", "text": f"candidate_id={c['id']}"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                base64.b64encode(Path(c["path"]).read_bytes()).decode()}}])
                    decision = parse_json_content(client.chat(model=model, messages=[{"role": "user", "content": content}],
                        max_tokens=1024, response_format={"type": "json_object"}))
                    selected = next((c for c in group if c["id"] == decision.get("candidate_id")), None)
                    if selected:
                        winners.append({**selected, "caption": decision.get("caption", ""), "reason": decision.get("reason", "")})
                if winners:
                    content = [{"type": "text", "text":
                        "Choose the best fully revealed, readable candidate supporting the request. "
                        "Return JSON candidate_id or null, caption, reason, and crop(optional normalized "
                        "[x,y,width,height]). Compare all winners; do not prefer later frames automatically.\n" +
                        json.dumps(request, ensure_ascii=False)}]
                    for c in winners:
                        content.extend([{"type": "text", "text": f"candidate_id={c['id']}"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                base64.b64encode(Path(c["path"]).read_bytes()).decode()}}])
                    final = parse_json_content(client.chat(model=model, messages=[{"role": "user", "content": content}],
                        max_tokens=1024, response_format={"type": "json_object"}))
                    winner = next((c for c in winners if c["id"] == final.get("candidate_id")), None)
                    if winner is None:
                        continue
                    original = winner["path"]
                    crop = final.get("crop")
                    if crop is not None:
                        if (not isinstance(crop, list) or len(crop) != 4 or
                            any(not isinstance(v, (int, float)) for v in crop)):
                            raise ValueError("invalid normalized crop")
                        x, y, width, height = crop
                        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
                            raise ValueError("crop exceeds frame")
                        cropped = directory / f"{key}-crop.jpg"
                        run([require_executable("ffmpeg"), "-y", "-i", original, "-vf",
                             f"crop=iw*{width}:ih*{height}:iw*{x}:ih*{y}", str(cropped)])
                        check = parse_json_content(client.chat(model=model, max_tokens=1024,
                            response_format={"type": "json_object"}, messages=[{"role": "user", "content": [
                                {"type": "text", "text": "Check cropped image against request; return JSON accepted(boolean), reason. " + json.dumps(request)},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(cropped.read_bytes()).decode()}}]}]))
                        if check.get("accepted") is not True:
                            continue
                        winner = {**winner, "path": str(cropped)}
                    result = {"status": "resolved", "figure_id": request["request_id"], "request_id": request["request_id"],
                              "start": min(request["start"], winner["timestamp"]),
                              "end": max(request["end"], winner["timestamp"] + 0.001), **winner,
                              "requested_interval": [request["start"], request["end"]],
                              "original_path": original, "crop": crop,
                              "caption": final.get("caption", winner["caption"]), "reason": final.get("reason", "")}
                    break
            write_json(cache, result)
        (resolved if result["status"] == "resolved" else unresolved).append(result)
    return {"figures": resolved, "unresolved": unresolved}
