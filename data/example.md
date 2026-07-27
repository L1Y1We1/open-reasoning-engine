# Reasoning Engine Sample Knowledge

The reasoning workflow has five stages: question analysis, evidence retrieval, answer generation,
evidence validation, and final output. When the validator finds that an answer lacks evidence or
does not address the question, the workflow can rewrite the retrieval query and search again once.

The default vector database is Qdrant. The default generation model is
`qwen2.5:7b-instruct`, served through Ollama, and the default embedding model is
`BAAI/bge-small-en-v1.5`. Every component can run locally, so documents do not need to be sent to
a third-party API.
