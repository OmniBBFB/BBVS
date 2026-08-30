from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import BBVSError

Message = dict[str, Any]


class ChatModel(Protocol):
    def chat(
        self, *, model: str, messages: list[Message], temperature: float = 0.0,
        max_tokens: int = 2048, response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str: ...


def _request_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BBVSError(f"推理请求失败 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BBVSError(f"推理请求失败: {exc}") from exc


def parse_json_content(content: str) -> Any:
    value = content.strip()
    if value.startswith("```"):
        first_newline, last_fence = value.find("\n"), value.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            value = value[first_newline + 1:last_fence].strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise BBVSError(f"模型未返回有效 JSON: {content[:500]}") from exc


@dataclass(slots=True)
class OpenAICompatibleClient:
    """HTTP adapter for OpenAI-compatible chat models."""
    base_url: str
    api_key: str = "EMPTY"
    timeout: float = 120.0

    def chat(
        self, *, model: str, messages: list[Message], temperature: float = 0.0,
        max_tokens: int = 2048, response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)
        result = _request_json(
            self.base_url.rstrip("/") + "/chat/completions", payload, self.api_key, self.timeout
        )
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BBVSError(f"模型返回格式异常: {result}") from exc

    def chat_json(self, **kwargs: Any) -> Any:
        kwargs.setdefault("response_format", {"type": "json_object"})
        return parse_json_content(self.chat(**kwargs))


@dataclass(slots=True)
class EmbeddingClient:
    base_url: str
    api_key: str = "EMPTY"
    timeout: float = 120.0

    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        result = _request_json(
            self.base_url.rstrip("/") + "/embeddings", {"model": model, "input": texts},
            self.api_key, self.timeout,
        )
        try:
            rows = sorted(result["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in rows]
        except (KeyError, TypeError) as exc:
            raise BBVSError(f"Embedding 返回格式异常: {result}") from exc


@dataclass(slots=True)
class RerankerClient:
    base_url: str
    api_key: str = "EMPTY"
    timeout: float = 120.0

    def rerank(
        self, *, model: str, query: str, documents: list[str], top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        payload: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        result = _request_json(
            self.base_url.rstrip("/") + "/rerank", payload, self.api_key, self.timeout
        )
        try:
            return [(item["index"], float(item["relevance_score"])) for item in result["results"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise BBVSError(f"Reranker 返回格式异常: {result}") from exc
