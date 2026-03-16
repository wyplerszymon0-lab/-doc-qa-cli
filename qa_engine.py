"""
qa_engine.py — retrieves relevant chunks and calls GPT to answer the question.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vector_store import VectorStore


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[str]     # source filenames used
    scores: list[float]    # cosine similarity scores


SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions strictly based on the provided context.
Rules:
- Answer only from the context below. Do not use outside knowledge.
- If the context does not contain the answer, say "I could not find an answer in the provided documents."
- Be concise and direct. Cite the source filename when relevant.
- If quoting directly, use quotation marks.
"""


def ask(
    question: str,
    store: "VectorStore",
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> Answer:
    # 1. retrieve relevant chunks
    results = store.query(question, top_k=top_k)

    # 2. build context block
    context_parts: list[str] = []
    sources: list[str] = []
    scores: list[float] = []

    for doc, score in results:
        context_parts.append(f"[Source: {doc.source}, chunk {doc.chunk_id}]\n{doc.text}")
        if doc.source not in sources:
            sources.append(doc.source)
        scores.append(score)

    context = "\n\n---\n\n".join(context_parts)

    # 3. call OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )

    answer_text = response.choices[0].message.content.strip()

    return Answer(
        question=question,
        answer=answer_text,
        sources=sources,
        scores=scores,
    )
