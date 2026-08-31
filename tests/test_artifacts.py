from bbvs.artifacts import variant


def test_variant_is_stable_and_ignores_credentials() -> None:
    first = variant("DeepSeek V4", {"model": "v4", "api_key": "secret-one"})
    second = variant("DeepSeek V4", {"model": "v4", "api_key": "secret-two"})

    assert first == second
    assert first.startswith("deepseek-v4--")


def test_variant_changes_when_result_configuration_changes() -> None:
    assert variant("model", {"model": "a"}) != variant("model", {"model": "b"})
