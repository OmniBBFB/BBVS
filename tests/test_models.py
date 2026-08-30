from bbvs.models import Frame, to_dict


def test_model_serialization_preserves_unicode() -> None:
    assert to_dict(Frame(path="一.jpg", timestamp=1.25, text=["你好"])) == {
        "path": "一.jpg",
        "timestamp": 1.25,
        "scene_score": None,
        "text": ["你好"],
    }
