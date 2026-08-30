from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

from .errors import CommandError, DependencyError


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise DependencyError(f"找不到 {name}，请先安装并确保它位于 PATH 中")
    return executable


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise CommandError(f"命令执行失败: {' '.join(command)}\n{detail}") from exc

