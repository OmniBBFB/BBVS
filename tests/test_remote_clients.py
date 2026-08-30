import io
import json
from unittest.mock import patch

from bbvs.llm import EmbeddingClient, RerankerClient, parse_json_content


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def test_parse_json_content_accepts_fenced_json() -> None:
    assert parse_json_content('```json\n{"ok": true}\n```') == {"ok": True}


def test_embedding_order_is_stable() -> None:
    body = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())):
        assert EmbeddingClient("http://localhost/v1").embed(model="m", texts=["a", "b"]) == [[1.0], [2.0]]


def test_reranker_returns_index_and_score() -> None:
    body = {"results": [{"index": 1, "relevance_score": 0.8}]}
    with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())):
        assert RerankerClient("http://localhost/v1").rerank(
            model="m", query="q", documents=["a", "b"]
        ) == [(1, 0.8)]
