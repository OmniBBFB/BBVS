import json

from bbvs.models import TimelineSegment
from bbvs.qa import answer_question
from bbvs.retrieval import SearchHit


class FakeChat:
    def chat(self, **kwargs):
        return json.dumps({"answer": "Interest rates.", "citations": [{"start": 0, "end": 60}], "insufficient_evidence": False})


def test_qa_returns_timestamped_answer() -> None:
    hits = [SearchHit(0, 0.9, TimelineSegment(0, 60, "Demand depends on rates."))]
    result = answer_question("What determines demand?", hits, FakeChat(), "m")
    assert result["citations"][0]["start"] == 0
