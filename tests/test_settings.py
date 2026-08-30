from bbvs.settings import AppSettings


def test_yaml_config_loads_complete_basic_pipeline(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""
runs_dir: output
services:
  llm: {base_url: http://llm/v1, model: model}
ocr: {engine: rapidocr, language: en}
asr: {engine: faster-whisper, model: tiny, device: cpu, compute_type: int8, language: en}
analysis: {verify: false, vision: false, summarize: true}
report: {enabled: true, filename: result.pdf, include_transcript: false, max_images: 0}
""", encoding="utf-8")
    settings = AppSettings.load(path)
    assert settings.runs_dir.name == "output"
    assert settings.ocr.language == "en"
    assert settings.asr.model == "tiny"
    assert settings.analysis.vision is False
    assert settings.services.vlm is None
    assert settings.report.filename == "result.pdf"


def test_yaml_config_rejects_non_mapping_section(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("services: []", encoding="utf-8")
    try:
        AppSettings.load(path)
    except ValueError as exc:
        assert "services" in str(exc)
    else:
        raise AssertionError("invalid config should fail")
