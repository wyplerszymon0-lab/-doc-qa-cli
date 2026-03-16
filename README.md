# doc-qa-cli 

A command-line RAG (Retrieval-Augmented Generation) tool that lets you ask natural language questions about your own documents — PDFs, TXT files, and Markdown — using OpenAI embeddings and GPT.

## Why it exists

LLMs don't know about your internal documents. This tool bridges that gap: it builds a local vector index from your files and uses semantic search to retrieve the most relevant chunks before sending them to GPT. No cloud vector database, no SaaS subscription.

## How it works

```
Documents (PDF/TXT/MD)
        │
        ▼
   Text chunking          ← 400-word chunks, 80-word overlap
        │
        ▼
  OpenAI Embeddings       ← text-embedding-3-small
        │
        ▼
 In-memory numpy array    ← saved to .pkl for reuse
        │
 ┌──────┘
 │  cosine similarity search (top-k)
 │
 ▼
Relevant chunks → GPT-4o-mini → Answer
```

## Quick start

```bash
git clone https://github.com/yourname/doc-qa-cli
cd doc-qa-cli
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# 1. Build the index (run once per document set)
python main.py index examples/ --save my_index.pkl

# 2. Ask a single question
python main.py ask "What is the refund policy?" --index my_index.pkl

# 3. Or start an interactive session
python main.py chat --index my_index.pkl
```

## Example session

```
$ python main.py chat --index my_index.pkl
[*] Interactive Q&A — 4 chunks loaded.
[*] Type 'exit' to quit.

You: What items cannot be refunded?

────────────────────────────────────────────────
Q: What items cannot be refunded?
────────────────────────────────────────────────
According to the policy (acme_policy.md), the following
items cannot be returned or refunded:
- Digital downloads and software licenses
- Perishable goods
- Custom-engraved or personalised items
- Gift cards

Sources: acme_policy.md
────────────────────────────────────────────────

You: How long do refunds take to process?
...
```

## Architecture

```
main.py                ← CLI with index / ask / chat sub-commands
src/
  loader.py            ← PDF, TXT, MD loaders + text chunking
  vector_store.py      ← numpy-based cosine similarity store
  qa_engine.py         ← retrieves chunks + calls GPT
examples/
  acme_policy.md       ← sample document
tests/
  test_loader.py       ← unit tests (no API key needed)
```

## Supported file types

| Extension | Parser |
|-----------|--------|
| `.txt`    | Built-in |
| `.md`     | Built-in (strips markdown) |
| `.pdf`    | pypdf |

## Running tests

```bash
python tests/test_loader.py
```

## Tech stack

- **Python 3.10+**
- **openai** — embeddings + chat completions
- **numpy** — vector similarity math
- **pypdf** — PDF text extraction
