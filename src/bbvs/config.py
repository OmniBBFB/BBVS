from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .io import read_json


@dataclass(frozen=True, slots=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 120.0


@dataclass(frozen=True, slots=True)
class Services:
    llm: Endpoint
    vlm: Endpoint | None = None
    embedding: Endpoint | None = None
    reranker: Endpoint | None = None

    @classmethod
    def load(cls, path: Path) -> Services:
        payload = read_json(path)

        def endpoint(name: str, *, required: bool = False) -> Endpoint | None:
            item = payload.get(name) or {}
            prefix = f"BBVS_{name.upper()}_"
            base_url = os.environ.get(prefix + "BASE_URL", item.get("base_url"))
            model = os.environ.get(prefix + "MODEL", item.get("model"))
            if not base_url or not model:
                if required:
                    raise ValueError(f"缺少必需的 {name} endpoint 配置")
                return None
            return Endpoint(
                base_url=base_url,
                model=model,
                api_key=os.environ.get(prefix + "API_KEY", item.get("api_key", "EMPTY")),
                timeout=float(os.environ.get(prefix + "TIMEOUT", item.get("timeout", 120))),
            )

        return cls(
            llm=endpoint("llm", required=True),  # type: ignore[arg-type]
            vlm=endpoint("vlm"), embedding=endpoint("embedding"), reranker=endpoint("reranker"),
        )
