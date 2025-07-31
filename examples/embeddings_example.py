"""
embeddings_example.py: CLI for embeddings and cosine similarity with ai-sdk.
Usage: embeddings_example.py --values VAL1 VAL2 [VAL3 ...] [--model MODEL_ID]
"""

import os
import argparse
from dotenv import load_dotenv

from ai_sdk import openai, embed_many, cosine_similarity  # type: ignore[attr-defined]

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Embeddings CLI using ai-sdk.")
    parser.add_argument(
        "--values", nargs="+", required=True, help="List of values to embed."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AI_SDK_EMBED_MODEL", "text-embedding-3-small"),
        help="Embedding model ID.",
    )
    args = parser.parse_args()

    model = openai.embedding(args.model, api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[attr-defined]
    res = embed_many(model=model, values=args.values)
    for i, val in enumerate(args.values):
        print(f"Value '{val}' embedding length: {len(res.embeddings[i])}")

    # Compute pairwise similarity for first two
    if len(res.embeddings) > 1:
        sim = cosine_similarity(res.embeddings[0], res.embeddings[1])
        print(f"Similarity between '{args.values[0]}' and '{args.values[1]}': {sim}")


if __name__ == "__main__":
    main()
