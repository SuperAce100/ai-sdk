import os
import math

import pytest

from ai_sdk import embed_many, embed  # type: ignore
from ai_sdk.embed import cosine_similarity
from ai_sdk.providers.openai import embedding as _embedding  # Import factory

# Skip real OpenAI calls if key missing
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)

MODEL_ID = os.getenv("AI_SDK_EMBED_MODEL", "text-embedding-3-small")

def _get_model():
    return _embedding(MODEL_ID)


def test_cosine_similarity_math():
    """Pure math check independent of provider."""
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_embed_many_and_single():
    model = _get_model()
    values = ["sunny day at the beach", "rainy afternoon in the city"]

    res_many = embed_many(model=model, values=values)

    # Basic sanity checks
    assert len(res_many.embeddings) == len(values)
    assert len(res_many.embeddings[0]) > 0

    # Single embedding via embed()
    res_single = embed(model=model, value=values[0])
    assert len(res_single.embedding) == len(res_many.embeddings[0])

    # Cosine similarity between identical embeddings from different calls should be high (>0.99)
    sim = cosine_similarity(res_single.embedding, res_many.embeddings[0])
    assert sim > 0.99

    # Cosine similarity between different sentences should be lower than identical (>0.3 for text data, just check not almost equal)
    sim_diff = cosine_similarity(res_many.embeddings[0], res_many.embeddings[1])
    assert sim_diff < 0.99
