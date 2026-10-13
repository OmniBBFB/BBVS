from copy import deepcopy
from pathlib import Path
import json

import pytest

from bbvs.teaching_document import validate, render_html
from bbvs.latex_renderer import export
from bbvs.document_review import review
from bbvs.models import EvidenceUnit


def document():
    return {"evidence_ids": ["k1"], "figures": [], "chapters": [{"title": "需求", "blocks": [
        {"id": "b1", "kind": "callout", "signal": "misconception", "evidence_ids": ["k1"], "text": "<script>"},
        {"id": "b2", "kind": "summary", "evidence_ids": ["k1"], "text": "需求会变。"}]}]}


def test_validate_rejects_unknown_figure_and_callout_without_evidence():
    d = document()
    d["chapters"][0]["blocks"][0] = {"id": "b1", "kind": "figure", "figure_id": "missing"}
    with pytest.raises(ValueError, match="figure"):
        validate(d)
    d = document()
    d["chapters"][0]["blocks"][0]["evidence_ids"] = []
    with pytest.raises(ValueError, match="evidence"):
        validate(d)


def test_html_and_tex_preserve_structure_and_escape(tmp_path):
    d = document()
    assert "&lt;script&gt;" in render_html(d)
    d["chapters"][0]["blocks"][0]["text"] = "a_b & 10%"
    output = export(d, tmp_path / "notes.tex")
    text = output.read_text()
    assert r"a\_b \& 10\%" in text
    assert r"\begin{tcolorbox}" in text
    assert not list(tmp_path.glob("*.pdf"))


def test_review_repairs_only_reported_block():
    class Client:
        def __init__(self):
            self.responses = iter([json.dumps({"issues": [{"block_id": "b1", "severity": "error", "reason": "unsupported"}]}),
                                   "已修复", json.dumps({"issues": []})])
        def chat(self, **kwargs):
            return next(self.responses)
    d = document()
    original = deepcopy(d["chapters"][0]["blocks"][1])
    result = review(d, [EvidenceUnit("u1", 0, 1, "需求会变")], Client(), "m")
    assert result["status"] == "complete"
    assert d["chapters"][0]["blocks"][0]["text"] == "已修复"
    assert d["chapters"][0]["blocks"][1] == original


def test_review_errors_after_budget_remain_partial():
    class Client:
        def chat(self, **kwargs):
            return json.dumps({"issues": [{"block_id": "b1", "severity": "error", "reason": "unresolved"}]})
    assert review(document(), [], Client(), "m", rounds=0)["status"] == "partial"
