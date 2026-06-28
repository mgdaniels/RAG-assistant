"""Basic checks that the package imports and the configuration is self-consistent."""
from ragassistant.config import settings


def test_settings_have_sensible_defaults():
    assert settings.top_k > 0
    assert settings.chunk_size > settings.chunk_overlap >= 0
    assert settings.collection_name
    assert str(settings.llm_base_url).startswith("http")
