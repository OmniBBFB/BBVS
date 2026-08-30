import json

import pytest

from bbvs.config import Services
from bbvs.llm import DeepSeekCompatibleClient, create_chat_client


def test_basic_services_only_requires_llm(tmp_path) -> None:
    path = tmp_path / "services.json"
    path.write_text(json.dumps({"llm": {"base_url": "http://llm/v1", "model": "m"}}))
    services = Services.load(path)
    assert services.llm.model == "m"
    assert services.vlm is None
    assert services.embedding is None
    assert services.reranker is None


def test_services_requires_llm(tmp_path) -> None:
    path = tmp_path / "services.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="llm"):
        Services.load(path)


def test_deepseek_provider_selects_deepseek_chat_adapter(tmp_path) -> None:
    path = tmp_path / "services.json"
    path.write_text(json.dumps({"llm": {
        "provider": "deepseek", "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash", "api_key": "local-secret",
    }}))

    endpoint = Services.load(path).llm

    assert endpoint.api_key == "local-secret"
    assert isinstance(create_chat_client(endpoint), DeepSeekCompatibleClient)
