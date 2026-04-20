#!/usr/bin/env python3
"""
Apply lexical / BM25 feature abstention on top of vanilla RAG JSON outputs.

Reads string-keyed JSON: typically ``squad_retrieval.json`` (retrieval-only,
from ``rag.py``) or full RAG outputs with ``response``. Adds ``decision``
(ANSWER/ABSTAIN), ``abstention`` (1 = abstained, 0 = answered). On abstain,
sets ``response`` to ``I do not know.`` for ``evaluation.load_rag_inference_json``.

**Training** (default corpus: ``squad_train.json``):

  python feature_gated_rag.py --fit
  python feature_gated_rag.py --fit squad_train.json --weights-out lexical_abstention_weights.json

  - SQuAD **array** JSON: labels from ``is_impossible``; passages = gold ``context``.

  - Inference **dict** with ``decision`` + ``question`` + ``retrieved_context`` (teacher).

**Apply** (trained ``LexicalAbstentionModel`` JSON required: ``--weights`` or
``lexical_abstention_weights.json`` in the working directory; default input
``squad_retrieval.json``):

  python feature_gated_rag.py --inputs squad_retrieval.json --outputs squad_feature_gated_outputs.json
  python feature_gated_rag.py --weights my_weights.json --inputs squad_rag_outputs.json --outputs out.json

When ``--rag-outputs`` points at vanilla RAG JSON (e.g. ``squad_rag_outputs.json``), answered rows
copy ``response`` from that file instead of leaving it empty (match by example key / ``ids`` / question).

Dense retrieval features use ``--embedding-model`` (default ``all-MiniLM-L6-v2``); pass ``--no-dense``
to train/apply with the dense block zeroed (smaller/faster, must match between fit and apply).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lexical_abstention_model import (
    DensePassageEncoder,
    LexicalAbstentionModel,
    train_from_labeled_examples,
)
from squad_utils import (
    RagOutputsResponseCache,
    gold_answer_label_for_squad_record,
    ordered_gold_answers_shortest_first,
)


def _sort_example_items(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    def key_fn(item: Tuple[str, Any]) -> Tuple[int, Any]:
        k = item[0]
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(data.items(), key=key_fn)


def _passages_from_example(ex: Dict[str, Any]) -> List[str]:
    ctx = ex.get("retrieved_context")
    if ctx is None:
        ctx = ex.get("retrieved_passages")
    if ctx is None:
        return []
    if isinstance(ctx, str):
        return [ctx] if ctx.strip() else []
    if isinstance(ctx, list):
        return [str(p) for p in ctx if p is not None and str(p).strip()]
    return []


def _squad_list_to_inference_dict(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Wrap a SQuAD JSON array as string-keyed inference examples (gold context as retrieved passages)."""
    out: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        ctx = rec.get("context") or []
        if isinstance(ctx, str):
            passages = [ctx] if str(ctx).strip() else []
        else:
            passages = [str(p) for p in ctx if p is not None and str(p).strip()]
        gold = gold_answer_label_for_squad_record(rec)
        ordered = (
            ordered_gold_answers_shortest_first(rec.get("answers") or [])
            if not rec.get("is_impossible")
            else []
        )
        out[str(i)] = {
            "question": str(rec.get("question", "")),
            "gold answer": gold,
            "gold_answers": ordered if ordered else (rec.get("answers") or []),
            "is_impossible": rec.get("is_impossible", False),
            "ids": rec.get("ids"),
            "response": "",
            "retrieved_context": passages,
        }
    return out


def _load_gating_input(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return _squad_list_to_inference_dict(data)
    raise ValueError(f"Unsupported JSON root for --inputs: {type(data).__name__}")


def _gold_answer_from_example(ex: Dict[str, Any]) -> str:
    for key in ("gold_answer", "gold answer"):
        v = ex.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    alts = ex.get("gold_answers")
    if isinstance(alts, list) and alts:
        return str(alts[0]).strip()
    return ""


def _normalize_gold_key(ex: Dict[str, Any]) -> None:
    if "gold_answer" not in ex or not str(ex.get("gold_answer", "")).strip():
        g = _gold_answer_from_example(ex)
        if g:
            ex["gold_answer"] = g


def _normalize_retrieval_key(ex: Dict[str, Any]) -> None:
    if "retrieved_context" not in ex and ex.get("retrieved_passages") is not None:
        ex["retrieved_context"] = ex["retrieved_passages"]


def _fit_examples_from_squad_records(
    records: List[Dict[str, Any]],
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build labeled rows for lexical abstention training from a SQuAD JSON array.

    Uses each example's ``context`` passages as the text seen by the feature
    extractor; ``is_impossible`` defines ABSTAIN vs ANSWER.
    """
    if max_examples is not None:
        records = records[: int(max_examples)]
    out: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ctx = rec.get("context") or []
        if isinstance(ctx, str):
            passages = [ctx] if str(ctx).strip() else []
        else:
            passages = [str(p) for p in ctx if p is not None and str(p).strip()]
        out.append(
            {
                "question": str(rec.get("question", "")),
                "retrieved_context": passages,
                "decision": "ABSTAIN" if rec.get("is_impossible") else "ANSWER",
            }
        )
    return out


def _load_fit_examples(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return _fit_examples_from_squad_records(data, max_examples=max_examples)
    if isinstance(data, dict):
        items = [ex for _, ex in _sort_example_items(data) if isinstance(ex, dict)]
        if max_examples is not None:
            items = items[: int(max_examples)]
        return items
    raise ValueError(f"Unsupported JSON root type for --fit: {type(data).__name__}")


def apply_feature_gating(
    data: Dict[str, Dict[str, Any]],
    model: LexicalAbstentionModel,
    *,
    threshold: float,
    response_cache: Optional[RagOutputsResponseCache] = None,
    dense_encoder: Optional[DensePassageEncoder] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, ex in _sort_example_items(data):
        if not isinstance(ex, dict):
            continue
        row = dict(ex)
        _normalize_gold_key(row)
        _normalize_retrieval_key(row)

        q = str(row.get("question", ""))
        passages = _passages_from_example(row)

        abstain, p_answer = model.should_abstain(
            q, passages, threshold=threshold, dense_encoder=dense_encoder
        )

        decision = "ABSTAIN" if abstain else "ANSWER"
        row["decision"] = decision
        row["abstention"] = 1 if abstain else 0
        row["lexical_gating_mode"] = "logistic"
        row["lexical_p_answer"] = round(p_answer, 6)
        row["justification"] = (
            f"Lexical gate (logistic): p(answer)={p_answer:.3f}; threshold={threshold}"
        )

        if abstain:
            row["response"] = "I do not know."
        elif response_cache is not None:
            cur = row.get("response")
            if cur is None or not str(cur).strip():
                hit = response_cache.lookup(
                    example_key=k, ids=row.get("ids"), question=q
                )
                if hit:
                    row["response"] = hit
        out[k] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-based abstention on RAG JSON outputs.")
    parser.add_argument(
        "--inputs",
        default="squad_retrieval.json",
        help="Retrieval JSON (dict, from rag.py) or full RAG outputs; or a SQuAD array (gold context as passages).",
    )
    parser.add_argument(
        "--outputs",
        default="squad_feature_gated_outputs.json",
        help="Where to write gated JSON.",
    )
    parser.add_argument(
        "--rag-outputs",
        default=os.getenv("SQUAD_RAG_OUTPUTS_JSON", "squad_rag_outputs.json"),
        help=(
            "Vanilla RAG JSON (string-keyed) to copy ``response`` for ANSWER rows when input "
            "has no response. Set to a non-existent path to skip. Default: env SQUAD_RAG_OUTPUTS_JSON "
            "or squad_rag_outputs.json."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Trained LexicalAbstentionModel JSON. Required for apply unless you use --fit "
            "(then defaults to --weights-out). If omitted, loads lexical_abstention_weights.json when present."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Abstain when P(ANSWER) is below this.",
    )
    parser.add_argument(
        "--fit",
        nargs="?",
        const="squad_train.json",
        default=None,
        metavar="PATH",
        help=(
            "Train weights: bare ``--fit`` uses squad_train.json. "
            "Accepts a SQuAD array (is_impossible → decision) or a dict of examples with ``decision``."
        ),
    )
    parser.add_argument(
        "--fit-max-examples",
        type=int,
        default=None,
        help="Optional cap on training rows (preserves file order).",
    )
    parser.add_argument(
        "--weights-out",
        default="lexical_abstention_weights.json",
        help="Path to save weights after --fit; same file is loaded for the apply step in that run.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer name for dense cosine features (train + apply when model uses dense).",
    )
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense encoder: dense block is all zeros (train and apply must match).",
    )
    args = parser.parse_args()

    if args.fit is not None:
        examples_list = _load_fit_examples(args.fit, max_examples=args.fit_max_examples)
        dense_enc: Optional[DensePassageEncoder] = None
        if not args.no_dense:
            dense_enc = DensePassageEncoder(args.embedding_model)
            print(f"[fit] Dense features: SentenceTransformer({args.embedding_model!r})")
        model = train_from_labeled_examples(
            examples_list,
            dense_encoder=dense_enc,
            use_dense_features=not args.no_dense,
        )
        Path(args.weights_out).write_text(
            json.dumps(model.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"Saved trained weights to {args.weights_out} ({len(examples_list)} training rows)")
        weights_path = args.weights_out
    else:
        weights_path = args.weights
        if weights_path is None:
            default_wp = Path("lexical_abstention_weights.json")
            if default_wp.is_file():
                weights_path = str(default_wp)
        if weights_path is None or not Path(weights_path).is_file():
            parser.error(
                "Trained weights are required: use --fit TRAIN.json (writes then loads "
                "--weights-out), or pass --weights PATH, or place lexical_abstention_weights.json "
                "in the current directory."
            )

    model = LexicalAbstentionModel.load(weights_path)
    print(f"Loaded logistic weights from {weights_path}")

    dense_apply: Optional[DensePassageEncoder] = None
    if model.use_dense_features and not args.no_dense:
        emb_name = model.embedding_model_name or args.embedding_model
        dense_apply = DensePassageEncoder(emb_name)
        print(f"[apply] Dense features: SentenceTransformer({emb_name!r})")
    elif model.use_dense_features and args.no_dense:
        print(
            "[apply] Warning: model was trained with dense features but --no-dense set; "
            "dense block will be zeros (distribution mismatch)."
        )

    data = _load_gating_input(args.inputs)
    response_cache = RagOutputsResponseCache.load(args.rag_outputs)
    if response_cache is not None:
        print(f"Will fill empty ANSWER responses from {args.rag_outputs!r}")
    gated = apply_feature_gating(
        data,
        model,
        threshold=args.threshold,
        response_cache=response_cache,
        dense_encoder=dense_apply,
    )

    Path(args.outputs).write_text(json.dumps(gated, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(gated)
    n_abs = sum(1 for ex in gated.values() if ex.get("abstention") == 1)
    print(f"Wrote {n} examples to {args.outputs} (abstained={n_abs}, answered={n - n_abs})")


if __name__ == "__main__":
    main()
