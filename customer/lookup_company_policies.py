import re

import numpy as np
import openai
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


def lookup_policy(query: str, retriever: VectorStoreRetriever) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events.

    Args:
        query (str): The query string to search for in the policies.
        retriever (VectorStoreRetriever): The retriever instance for finding relevant documents.

    Returns:
        str: A string containing the content of the retrieved policy documents, joined by newlines.
    """

    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])


def get_retriever():
    response = requests.get(
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
    )
    response.raise_for_status()
    faq_text = response.text

    docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

    retriever = VectorStoreRetriever.from_docs(docs, openai.Client())

    return retriever


if __name__ == "__main__":
    load_dotenv()
    retriever = get_retriever()
    policy_text = lookup_policy("Can I change my flight date?", retriever)
    print(policy_text)
