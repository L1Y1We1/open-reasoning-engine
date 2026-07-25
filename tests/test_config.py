from reasoning_engine.config import AppSettings


def test_empty_secret_values_become_none() -> None:
    value = AppSettings(_env_file=None, qdrant_api_key="", milvus_token="")
    assert value.qdrant_api_key is None
    assert value.milvus_token is None


def test_qdrant_is_default() -> None:
    value = AppSettings(_env_file=None)
    assert value.vector_store == "qdrant"
    assert value.embedding_dimension == 512

