import io
import json
from unittest.mock import patch

from bbvs.llm import DeepSeekCompatibleClient, OpenAICompatibleClient


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


def test_deepseek_client_translates_non_thinking_parameter() -> None:
    body = {"choices": [{"message": {"content": '{"ok":true}'}}]}
    with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())) as call:
        result = DeepSeekCompatibleClient("https://api.deepseek.com", "secret").chat(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    request = call.call_args.args[0]
    payload = json.loads(request.data)
    assert result == '{"ok":true}'
    assert request.full_url == "https://api.deepseek.com/chat/completions"
    assert payload["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in payload


def test_deepseek_client_translates_reasoning_effort() -> None:
    body = {"choices": [{"message": {"content": "{}"}}]}
    with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())) as call:
        DeepSeekCompatibleClient("https://api.deepseek.com", "secret").chat(
            model="deepseek-v4-pro", messages=[{"role": "user", "content": "Extract JSON"}],
            extra_body={"reasoning_effort": "low"},
        )

    payload = json.loads(call.call_args.args[0].data)
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
