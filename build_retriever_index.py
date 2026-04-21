"""
Build a persisted FAISS + SentenceTransformer retriever index for live query-time retrieval.

By default merges **unique** gold ``context`` passages from both ``squad_train.json`` and
``squad_test.json`` (order preserved: train first, then test-only passages). Override with
``--squad-json`` (repeatable).

Output directory (positional) receives ``faiss.index``, ``documents.json``, ``meta.json``
(see :meth:`rag.DocumentRetriever.save_index`), plus ``sources.json`` listing input paths.

Dependencies match the project (see ``requirements.txt``): ``sentence-transformers``,
``faiss-cpu``, ``numpy``, ``torch``, etc.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from rag import DocumentRetriever, load_squad_passages_union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write the index (created if missing).",
    )
    parser.add_argument(
        "--squad-json",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "SQuAD JSON array (passages in ``context``). Repeat to merge multiple files. "
            "Default: squad_train.json and squad_test.json when both exist."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint name (stored in meta.json).",
    )
    args = parser.parse_args()

    paths: List[str]
    if args.squad_json:
        paths = list(args.squad_json)
    else:
        candidates = ["squad_train.json", "squad_test.json"]
        paths = [p for p in candidates if os.path.isfile(p)]
        if not paths:
            parser.error(
                "No squad_train.json or squad_test.json in the current directory; "
                "pass one or more --squad-json PATH."
            )

    docs = load_squad_passages_union(paths)
    if not docs:
        parser.error("No passages extracted; check SQuAD ``context`` fields.")

    retriever = DocumentRetriever(model_name=args.embedding_model)
    print(f"Chunking + embedding with {args.embedding_model!r} …")
    retriever.add_documents(docs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save_index(str(out_dir))

    sources_path = out_dir / "sources.json"
    sources_path.write_text(
        json.dumps({"squad_json": paths}, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved index to {out_dir.resolve()!s} "
        f"({retriever.index.ntotal} vectors, model={retriever.model_name!r})"
    )
    print(f"Wrote {sources_path.name} with {len(paths)} source path(s).")


if __name__ == "__main__":
    main()
