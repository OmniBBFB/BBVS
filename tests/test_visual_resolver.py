import json
from types import SimpleNamespace

from bbvs import visual_resolver as vr


def test_resolver_samples_media_compares_winners_and_reuses_cache(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(vr, "probe", lambda p: SimpleNamespace(duration=30))
    monkeypatch.setattr(vr, "require_executable", lambda n: n)
    calls = []
    def run(args):
        calls.append(args)
        from pathlib import Path
        Path(args[-1]).write_bytes(b"jpeg")
    monkeypatch.setattr(vr, "run", run)
    class Client:
        def __init__(self): self.calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            return json.dumps({"candidate_id": 0, "caption": "完整图", "reason": "可读"})
    client = Client()
    request = {"request_id": "r1", "start": 10, "end": 12, "required_content": ["图例"]}
    result = vr.resolve([request], video, tmp_path / "frames", client, "m", max_candidates=4)
    assert result["figures"][0]["timestamp"] == 2
    assert result["figures"][0]["requested_interval"] == [10, 12]
    assert result["figures"][0]["start"] <= result["figures"][0]["timestamp"] < result["figures"][0]["end"]
    assert len(calls) == 4
    assert client.calls == 2
    again = vr.resolve([request], video, tmp_path / "frames", client, "m", max_candidates=4)
    assert again == result and client.calls == 2


def test_no_acceptable_frame_is_explicitly_unresolved(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(vr, "probe", lambda p: SimpleNamespace(duration=30))
    monkeypatch.setattr(vr, "require_executable", lambda n: n)
    def run(args):
        from pathlib import Path
        Path(args[-1]).write_bytes(b"jpeg")
    monkeypatch.setattr(vr, "run", run)
    class Client:
        def chat(self, **kwargs): return '{"candidate_id":null}'
    result = vr.resolve([{"request_id": "r", "start": 10, "end": 12}], video,
                        tmp_path / "frames", Client(), "m", attempts=1, max_candidates=2)
    assert not result["figures"] and result["unresolved"][0]["status"] == "unresolved"
