#!/usr/bin/env python3
"""
doc-qa-cli — ask questions about your own documents.

Commands:
  index   Build a vector index from document files.
  ask     Ask a single question (non-interactive).
  chat    Start an interactive Q&A session.

Examples:
  python main.py index docs/ --save my_index.pkl
  python main.py ask "What is the refund policy?" --index my_index.pkl
  python main.py chat --index my_index.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> int:
    from src.loader import load_file, load_directory
    from src.vector_store import VectorStore

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Path not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[*] Loading documents from {input_path} ...")
    docs = load_directory(input_path) if input_path.is_dir() else load_file(input_path)

    if not docs:
        print("[error] No documents loaded.", file=sys.stderr)
        return 1

    print(f"[*] Loaded {len(docs)} chunks from {len({d.source for d in docs})} file(s).")

    store = VectorStore(model=args.model)
    store.build(docs, show_progress=True)

    save_path = Path(args.save) if args.save else input_path.parent / "index.pkl"
    store.save(save_path)
    print(f"[*] Done. Index saved to {save_path}")
    return 0


def _load_store(index_path: str) -> "VectorStore":
    from src.vector_store import VectorStore
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    return VectorStore.load(path)


def _print_answer(answer: "Answer") -> None:
    print(f"\n{'─'*60}")
    print(f"Q: {answer.question}")
    print(f"{'─'*60}")
    print(answer.answer)
    print(f"\n📄 Sources: {', '.join(answer.sources)}")
    print(f"{'─'*60}\n")


def cmd_ask(args: argparse.Namespace) -> int:
    from src.qa_engine import ask

    store = _load_store(args.index)
    answer = ask(args.question, store, top_k=args.top_k, model=args.model)
    _print_answer(answer)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from src.qa_engine import ask

    store = _load_store(args.index)
    print(f"\n[*] Interactive Q&A session. {len(store)} chunks loaded.")
    print("[*] Type 'exit' or Ctrl-C to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Goodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("[*] Goodbye.")
            break

        try:
            answer = ask(question, store, top_k=args.top_k, model=args.model)
            _print_answer(answer)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-qa",
        description="Ask questions about your own documents using OpenAI embeddings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Build vector index from documents")
    p_index.add_argument("input", help="Path to a file or directory")
    p_index.add_argument("--save", default=None, help="Where to save the index .pkl (default: <input>/index.pkl)")
    p_index.add_argument("--model", default="text-embedding-3-small", help="Embedding model")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", help="The question to ask")
    p_ask.add_argument("--index", required=True, help="Path to the .pkl index file")
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    p_ask.add_argument("--model", default="gpt-4o-mini", help="Chat model to use")

    # chat
    p_chat = sub.add_parser("chat", help="Start interactive Q&A session")
    p_chat.add_argument("--index", required=True, help="Path to the .pkl index file")
    p_chat.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    p_chat.add_argument("--model", default="gpt-4o-mini", help="Chat model to use")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {"index": cmd_index, "ask": cmd_ask, "chat": cmd_chat}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
