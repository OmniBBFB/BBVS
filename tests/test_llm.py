import io
import json
from unittest.mock import patch

from bbvs.llm import OpenAICompatibleClient


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_openai_compatible_client_builds_v1_request() -> None:
    body = {"choices": [{"message": {"content": "pong"}}]}
    with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())) as call:
        result = OpenAICompatibleClient("http://localhost:8000/v1/").chat(
            model="demo", messages=[{"role": "user", "content": "ping"}]
        )
    assert result == "pong"
    request = call.call_args.args[0]
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert json.loads(request.data)["model"] == "demo"
    assert json.loads(request.data)["max_tokens"] == 2048
