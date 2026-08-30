import json

import pytest

from bbvs.config import Services


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
