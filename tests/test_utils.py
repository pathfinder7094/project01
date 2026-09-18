from app.utils.slugs import stable_slug


def test_korean_slug() -> None:
    assert stable_slug("멀티 헤드 Attention") == "멀티-헤드-attention"
