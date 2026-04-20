import argparse
import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import evaluate
from scipy import stats

_BERTSCORE = evaluate.load("bertscore")

# Default judge weights (match epistemic_gated_rag.EpistemicJudge)
_EPISTEMIC_WEIGHTS = (0.35, 0.45, 0.20)




# =========================
# Text helpers
# =========================

def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Optional[str]) -> List[str]:
    return normalize_text(text).split()


def as_chunk_list(context: Union[str, List[str], None]) -> List[str]:
    if context is None:
        return []
    if isinstance(context, list):
        return context
    return [context]


def join_chunks(context: Union[str, List[str], None]) -> str:
    return "\n".join(as_chunk_list(context))


# =========================
# Inference JSON loading (vanilla RAG + epistemic gated)
# =========================


def _sort_example_items(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    def key_fn(item: Tuple[str, Any]) -> Tuple[int, Union[int, str]]:
        k = item[0]
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(data.items(), key=key_fn)


def gold_answer_from_example(ex: Dict[str, Any]) -> str:
    for key in ("gold_answer", "gold answer"):
        v = ex.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    alts = ex.get("gold_answers")
    if isinstance(alts, list) and alts:
        return str(alts[0]).strip()
    return ""


def retrieved_context_from_example(ex: Dict[str, Any]) -> Union[str, List[str], None]:
    return ex.get("retrieved_context") if ex.get("retrieved_context") is not None else ex.get(
        "retrieved_passages"
    )


def epistemic_numeric_score(ex: Dict[str, Any]) -> Tuple[float, str]:
    """
    Map judge output to a single numeric score for correlation with retrieval quality.

    - Structured mode: composite_score on 0–10 when present; else weighted blend of
      relevance/sufficiency/consistency when those are present.
    - Otherwise (minimal yes/no): 1.0 if decision is ANSWER else 0.0.

    Returns (score_on_0_to_1_scale, score_source).
    """
    comp = ex.get("composite_score")
    if comp is not None and not (isinstance(comp, float) and math.isnan(comp)):
        return float(comp) / 10.0, "structured_composite"

    rel = ex.get("relevance_score")
    suf = ex.get("sufficiency_score")
    con = ex.get("consistency_score")
    if rel is not None and suf is not None and con is not None:
        w_r, w_s, w_c = _EPISTEMIC_WEIGHTS
        blended = w_r * float(rel) + w_s * float(suf) + w_c * float(con)
        return blended / 10.0, "structured_dims_weighted"

    dec = str(ex.get("decision", "")).upper()
    return (1.0 if dec == "ANSWER" else 0.0), "binary_decision"


def is_gated_example(ex: Dict[str, Any]) -> bool:
    return "decision" in ex


def load_rag_inference_json(
    path: str,
) -> Tuple[List[str], List[str], List[str], List[Union[str, List[str], None]], List[Dict[str, Any]]]:
    """
    Load RAG / gated JSON (string-keyed dict of examples), e.g. squad_rag_outputs.json.

    Examples are ordered by numeric key when possible. Returns parallel lists plus
    raw example dicts (for epistemic fields and optional ids → gold context join).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object of examples, got {type(data).__name__}")

    questions: List[str] = []
    gold_answers: List[str] = []
    predicted_answers: List[str] = []
    retrieved_contexts: List[Union[str, List[str], None]] = []
    raw_examples: List[Dict[str, Any]] = []

    for _, ex in _sort_example_items(data):
        if not isinstance(ex, dict):
            continue
        questions.append(str(ex.get("question", "")))
        gold_answers.append(gold_answer_from_example(ex))
        pred = ex.get("response")
        predicted_answers.append("" if pred is None else str(pred))
        retrieved_contexts.append(retrieved_context_from_example(ex))
        raw_examples.append(ex)

    return questions, gold_answers, predicted_answers, retrieved_contexts, raw_examples


def load_squad_id_to_context(json_path: str) -> Dict[str, List[str]]:
    """Build ids → gold context passages from SQuAD-style JSON (same schema as rag.load_squad_json_records)."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected JSON array in {json_path}")
    out: Dict[str, List[str]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rid = rec.get("ids")
        if not rid:
            continue
        ctx = rec.get("context") or []
        if isinstance(ctx, list):
            out[str(rid)] = [str(p) for p in ctx if p is not None and str(p).strip()]
        else:
            out[str(rid)] = [str(ctx)]
    return out


def align_gold_contexts_to_examples(
    raw_examples: List[Dict[str, Any]],
    id_to_context: Dict[str, List[str]],
) -> List[Optional[List[str]]]:
    """One gold passage list per example, matched by ``ids`` when possible."""
    aligned: List[Optional[List[str]]] = []
    for ex in raw_examples:
        rid = ex.get("ids")
        if rid is None:
            aligned.append(None)
            continue
        key = str(rid).strip()
        aligned.append(id_to_context.get(key))
    return aligned


def correlation_pearson_spearman(
    x: Sequence[float],
    y: Sequence[float],
) -> Dict[str, Any]:
    """Pairwise correlation; drops non-finite values."""
    pairs: List[Tuple[float, float]] = []
    for a, b in zip(x, y):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fa) and math.isfinite(fb):
            pairs.append((fa, fb))
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    return {
        "n": len(pairs),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# =========================
# Answer metrics
# =========================

def token_precision_recall_f1(predicted: str, gold: str) -> Dict[str, float]:
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = pred_counter & gold_counter
    num_same = sum(overlap.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_bertscores(
    predicted_answers: Sequence[str],
    gold_answers: Sequence[str],
    lang: str = "en",
) -> List[Dict[str, float]]:
    result = _BERTSCORE.compute(
        predictions=list(predicted_answers),
        references=list(gold_answers),
        lang=lang,
    )
    return [
        {
            "precision": result["precision"][i],
            "recall": result["recall"][i],
            "f1": result["f1"][i],
        }
        for i in range(len(predicted_answers))
    ]


# =========================
# Retrieval / grounding metrics
# =========================

def context_answer_token_recall(
    retrieved_context: Union[str, List[str], None],
    gold_answer: str,
) -> float:
    context_text = join_chunks(retrieved_context)

    gold_tokens = tokenize(gold_answer)
    context_tokens = tokenize(context_text)

    if not gold_tokens:
        return 1.0

    gold_counter = Counter(gold_tokens)
    context_counter = Counter(context_tokens)
    overlap = gold_counter & context_counter
    covered = sum(overlap.values())

    return covered / sum(gold_counter.values())


def answer_faithfulness(
    predicted_answer: str,
    retrieved_context: Union[str, List[str], None],
) -> Dict[str, float]:
    """
    Simple groundedness / faithfulness proxy:
    fraction of predicted-answer tokens that are supported by retrieved context.
    """
    pred_tokens = tokenize(predicted_answer)
    context_tokens = tokenize(join_chunks(retrieved_context))

    if not pred_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    pred_counter = Counter(pred_tokens)
    context_counter = Counter(context_tokens)
    overlap = pred_counter & context_counter
    supported = sum(overlap.values())

    precision = supported / len(pred_tokens)

    # "Recall" here is context-coverage relative to supported claim tokens.
    # This is less important than precision for faithfulness, but included
    # for symmetry and convenience.
    if not context_tokens:
        recall = 0.0
    else:
        recall = supported / len(context_tokens)

    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def retrieval_precision_recall_f1(
    retrieved_chunks: Sequence[str],
    gold_chunks: Sequence[str],
) -> Dict[str, float]:
    retrieved_norm = {normalize_text(chunk) for chunk in retrieved_chunks if normalize_text(chunk)}
    gold_norm = {normalize_text(chunk) for chunk in gold_chunks if normalize_text(chunk)}

    if not retrieved_norm and not gold_norm:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not retrieved_norm or not gold_norm:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(retrieved_norm & gold_norm)
    precision = tp / len(retrieved_norm)
    recall = tp / len(gold_norm)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# =========================
# Main evaluation
# =========================

def _validate_equal_lengths(**named_lists: Sequence[Any]) -> int:
    lengths = {name: len(values) for name, values in named_lists.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"All inputs must have the same length. Got: {lengths}")
    return next(iter(unique_lengths))


def evaluate_rag_batch(
    questions: List[str],
    gold_answers: List[str],
    predicted_answers: List[str],
    retrieved_contexts: List[Union[str, List[str], None]],
    gold_contexts_list: Optional[List[Optional[List[str]]]] = None,
    compute_semantic_metrics: bool = True,
    bertscore_lang: str = "en",
    epistemic_raw_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    n = _validate_equal_lengths(
        questions=questions,
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        retrieved_contexts=retrieved_contexts,
    )

    if gold_contexts_list is None:
        gold_contexts_list = [None] * n

    _validate_equal_lengths(
        questions=questions,
        gold_contexts_list=gold_contexts_list,
    )

    if epistemic_raw_examples is not None:
        if len(epistemic_raw_examples) != n:
            raise ValueError(
                f"epistemic_raw_examples length {len(epistemic_raw_examples)} must match batch size {n}"
            )

    bertscores: List[Optional[Dict[str, float]]] = [None] * n
    if compute_semantic_metrics:
        bertscores = compute_bertscores(
            predicted_answers,
            gold_answers,
            lang=bertscore_lang,
        )

    results: List[Dict[str, Any]] = []

    for i in range(n):
        retrieved_chunks = as_chunk_list(retrieved_contexts[i])
        context_text = join_chunks(retrieved_contexts[i])

        lexical = token_precision_recall_f1(predicted_answers[i], gold_answers[i])
        faithfulness = answer_faithfulness(predicted_answers[i], retrieved_contexts[i])

        result: Dict[str, Any] = {
            "question": questions[i],
            "gold_answer": gold_answers[i],
            "predicted_answer": predicted_answers[i],
            "retrieved_context": retrieved_contexts[i],
            "answer_metrics": {
                "token_precision": lexical["precision"],
                "token_recall": lexical["recall"],
                "token_f1": lexical["f1"],
            },
            "retrieval_metrics": {
                "answer_token_recall_in_context": context_answer_token_recall(
                    retrieved_contexts[i],
                    gold_answers[i],
                ),
            },
            "grounding_metrics": {
                "faithfulness_precision": faithfulness["precision"],
                "faithfulness_recall": faithfulness["recall"],
                "faithfulness_f1": faithfulness["f1"],
            },
        }
        
        #ADD BERT SCORES
        if bertscores[i] is not None:
            result["answer_metrics"]["bertscore_precision"] = bertscores[i]["precision"]
            result["answer_metrics"]["bertscore_recall"] = bertscores[i]["recall"]
            result["answer_metrics"]["bertscore_f1"] = bertscores[i]["f1"]

        #ADD RETRIEVAL CHUNK METRICS IF GOLD CONTEXTS PROVIDED
        gold_contexts = gold_contexts_list[i]
        retrieval_results =  retrieval_precision_recall_f1(
                retrieved_chunks=retrieved_chunks,
                gold_chunks=gold_contexts,
            ) if gold_contexts is not None else None
        
        result["retrieval_metrics"]["chunk_precision"] = retrieval_results["precision"] if retrieval_results else None
        result["retrieval_metrics"]["chunk_recall"] = retrieval_results["recall"] if retrieval_results else None
        result["retrieval_metrics"]["chunk_f1"] = retrieval_results["f1"] if retrieval_results else None

        atr = result["retrieval_metrics"]["answer_token_recall_in_context"]
        result["retrieval_metrics"]["retrieval_accuracy_primary"] = atr

        if epistemic_raw_examples is not None:
            raw = epistemic_raw_examples[i]
            if is_gated_example(raw):
                score_0_1, score_src = epistemic_numeric_score(raw)
                result["epistemic_metrics"] = {
                    "score_0_1": score_0_1,
                    "score_source": score_src,
                    "decision": raw.get("decision"),
                    "composite_score": raw.get("composite_score"),
                    "relevance_score": raw.get("relevance_score"),
                    "sufficiency_score": raw.get("sufficiency_score"),
                    "consistency_score": raw.get("consistency_score"),
                }

        results.append(result)

    return results


def epistemic_retrieval_correlation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Correlation between epistemic judge score (0–1) and retrieval accuracy
    (answer-token recall in retrieved passages, 0–1).

    Only includes examples that have ``epistemic_metrics``.
    """
    xs: List[float] = []
    ys: List[float] = []
    score_sources: List[str] = []
    for r in results:
        em = r.get("epistemic_metrics")
        if not em:
            continue
        xs.append(float(em["score_0_1"]))
        ys.append(float(r["retrieval_metrics"]["answer_token_recall_in_context"]))
        score_sources.append(str(em.get("score_source", "")))

    if not xs:
        return {
            "n_examples": 0,
            "note": "No gated examples with epistemic_metrics in results.",
        }

    corr = correlation_pearson_spearman(xs, ys)
    src_counts = dict(Counter(score_sources))
    return {
        "n_examples": len(xs),
        "epistemic_score_sources": src_counts,
        "description": (
            "Pearson/Spearman between epistemic score (0–1) and answer-token recall in "
            "retrieved context (higher = gold answer better covered by retrieval)."
        ),
        **corr,
        "chunk_overlap_correlation": _optional_chunk_correlation(results),
    }


def _optional_chunk_correlation(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """If chunk_f1 is available, correlate epistemic score with chunk F1."""
    xs: List[float] = []
    ys: List[float] = []
    for r in results:
        em = r.get("epistemic_metrics")
        cf1 = r["retrieval_metrics"].get("chunk_f1")
        if not em or cf1 is None:
            continue
        xs.append(float(em["score_0_1"]))
        ys.append(float(cf1))
    if len(xs) < 2:
        return None
    out = correlation_pearson_spearman(xs, ys)
    out["metric"] = "chunk_f1"
    return out


# =========================
# Summary
# =========================

def summarize_rag_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
        if not valid:
            return None
        return sum(valid) / len(valid)

    summary: Dict[str, Any] = {
        "answer_metrics": {
            "token_precision": mean([r["answer_metrics"]["token_precision"] for r in results]),
            "token_recall": mean([r["answer_metrics"]["token_recall"] for r in results]),
            "token_f1": mean([r["answer_metrics"]["token_f1"] for r in results]),
            "bertscore_precision": mean([
                r["answer_metrics"].get("bertscore_precision") for r in results
            ]),
            "bertscore_recall": mean([
                r["answer_metrics"].get("bertscore_recall") for r in results
            ]),
            "bertscore_f1": mean([
                r["answer_metrics"].get("bertscore_f1") for r in results
            ]),
        },
        "retrieval_metrics": {
            "answer_token_recall_in_context": mean([
                r["retrieval_metrics"]["answer_token_recall_in_context"]
                for r in results
            ]),
            "chunk_precision": mean([
                r["retrieval_metrics"]["chunk_precision"]
                for r in results
            ]),
            "chunk_recall": mean([
                r["retrieval_metrics"]["chunk_recall"]
                for r in results
            ]),
            "chunk_f1": mean([
                r["retrieval_metrics"]["chunk_f1"]
                for r in results
            ]),
        },
        "grounding_metrics": {
            "faithfulness_precision": mean([
                r["grounding_metrics"]["faithfulness_precision"] for r in results
            ]),
            "faithfulness_recall": mean([
                r["grounding_metrics"]["faithfulness_recall"] for r in results
            ]),
            "faithfulness_f1": mean([
                r["grounding_metrics"]["faithfulness_f1"] for r in results
            ]),
        },
    }

    if any("epistemic_metrics" in r for r in results):
        summary["epistemic_vs_retrieval"] = epistemic_retrieval_correlation_summary(results)

    return summary


def load_inference_json(path: str) -> Tuple[List[str], List[str], List[str], List[Union[str, List[str], None]]]:
    """Backward-compatible loader (no raw examples). Prefer :func:`load_rag_inference_json`."""
    q, g, p, r, _ = load_rag_inference_json(path)
    return q, g, p, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG or gated RAG JSON outputs.")
    parser.add_argument(
        "--outputs",
        default="squad_rag_outputs.json",
        help="Path to RAG or gated JSON (e.g. squad_rag_outputs.json, squad_gated_outputs.json).",
    )
    parser.add_argument(
        "--squad-json",
        default=None,
        help="Optional SQuAD records JSON for gold passages (chunk overlap metrics via ids).",
    )
    parser.add_argument("--no-bert", action="store_true", help="Skip BERTScore (faster).")
    parser.add_argument("--print-per-example", action="store_true", help="Print full per-example dicts.")
    args = parser.parse_args()

    questions, gold_answers, predicted_answers, retrieved_contexts, raw_examples = load_rag_inference_json(
        args.outputs
    )

    gold_contexts_list: Optional[List[Optional[List[str]]]] = None
    if args.squad_json:
        id_to_ctx = load_squad_id_to_context(args.squad_json)
        gold_contexts_list = align_gold_contexts_to_examples(raw_examples, id_to_ctx)

    has_gated = any(is_gated_example(ex) for ex in raw_examples)
    epistemic_raw = raw_examples if has_gated else None

    results = evaluate_rag_batch(
        questions=questions,
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        retrieved_contexts=retrieved_contexts,
        gold_contexts_list=gold_contexts_list,
        compute_semantic_metrics=not args.no_bert,
        bertscore_lang="en",
        epistemic_raw_examples=epistemic_raw,
    )

    summary = summarize_rag_results(results)

    if args.print_per_example:
        print("\nPer-example results:")
        for result in results:
            print(result)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))