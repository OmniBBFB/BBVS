from __future__ import annotations

import re
from pathlib import Path

from .models import Frame
from .process import require_executable, run

_PTS_ONLY = re.compile(r"pts_time:(?P<time>-?[0-9.]+)")


def extract_keyframes(
    video: Path,
    output_dir: Path,
    threshold: float = 0.3,
    max_frames: int = 500,
) -> list[Frame]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("scene threshold 必须在 0 到 1 之间")
    ffmpeg = require_executable("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    filter_graph = f"select='eq(n,0)+gt(scene,{threshold})',metadata=print,showinfo"
    result = run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            filter_graph,
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            "-y",
            str(pattern),
        ]
    )
    timestamps: list[float] = []
    for line in result.stderr.splitlines():
        match = _PTS_ONLY.search(line) if "showinfo" in line else None
        if match:
            timestamps.append(float(match.group("time")))
    paths = sorted(output_dir.glob("frame_*.jpg"))
    return [
        Frame(path=str(path.resolve()), timestamp=timestamps[index] if index < len(timestamps) else 0.0)
        for index, path in enumerate(paths)
    ]
