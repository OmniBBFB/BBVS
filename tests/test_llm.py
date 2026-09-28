import io
import json
from unittest.mock import patch

from bbvs.errors import BBVSError
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


def test_client_reduces_output_budget_when_prompt_nearly_fills_context() -> None:
    error = BBVSError(
        "推理请求失败 HTTP 400: This model's maximum context length is 8192 tokens. "
        "However, you requested 4096 output tokens and your prompt contains at least "
        "4097 input tokens, for a total of at least 8193 tokens."
    )
    body = {"choices": [{"message": {"content": "completed"}}]}

    with patch("bbvs.llm._request_json", side_effect=[error, body]) as request:
        result = OpenAICompatibleClient("http://localhost:8000/v1").chat(
            model="demo", messages=[{"role": "user", "content": "long prompt"}],
            max_tokens=4096,
        )

    assert result == "completed"
    assert request.call_count == 2
    assert request.call_args_list[0].args[1]["max_tokens"] == 4096
    assert request.call_args_list[1].args[1]["max_tokens"] == 3839


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


def test_deepseek_structured_output_disables_thinking_by_default() -> None:
    body = {"choices": [{"message": {"content": '{"ok":true}'}}]}
    with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())) as call:
        DeepSeekCompatibleClient("https://api.deepseek.com", "secret").chat(
            model="deepseek-v4-flash", messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

    payload = json.loads(call.call_args.args[0].data)
    assert payload["thinking"] == {"type": "disabled"}


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
