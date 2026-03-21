import re
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union
import evaluate

_BERTSCORE = evaluate.load("bertscore")




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

        results.append(result)

    return results



# =========================
# Summary
# =========================

def summarize_rag_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return sum(valid) / len(valid)

    summary = {
        "answer_metrics": {
            "token_precision": mean([r["answer_metrics"]["token_precision"] for r in results]),
            "token_recall": mean([r["answer_metrics"]["token_recall"] for r in results]),
            "token_f1": mean([r["answer_metrics"]["token_f1"] for r in results]),
            "bertscore_precision": mean([
                r["answer_metrics"]["bertscore_precision"]
                for r in results
            ]),
            "bertscore_recall": mean([
                r["answer_metrics"]["bertscore_recall"]
                for r in results
            ]),
            "bertscore_f1": mean([
                r["answer_metrics"]["bertscore_f1"]
                for r in results
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

    chunk_scores = [
        r["retrieval_metrics"]["chunk_precision_recall_f1"]
        for r in results
        if "chunk_precision_recall_f1" in r["retrieval_metrics"]
    ]
    if chunk_scores:
        summary["retrieval_metrics"]["chunk_precision"] = mean([x["precision"] for x in chunk_scores])
        summary["retrieval_metrics"]["chunk_recall"] = mean([x["recall"] for x in chunk_scores])
        summary["retrieval_metrics"]["chunk_f1"] = mean([x["f1"] for x in chunk_scores])

    return summary


def load_inference_jsonl(path):
    questions = []
    gold_answers = []
    predicted_answers = []
    retrieved_contexts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            questions.append(ex["question"])
            gold_answers.append(ex["gold_answer"])
            predicted_answers.append(ex["predicted_answer"])
            retrieved_contexts.append(ex["retrieved_context"])

    return questions, gold_answers, predicted_answers, retrieved_contexts


if __name__ == "__main__":
    
    #replace "inference_results.jsonl" with avi's output file
    # questions, gold_answers, predicted_answers, retrieved_contexts = load_inference_jsonl("inference_results.jsonl")
    questions = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
    ]

    gold_answers = [
        "Paris is the capital of France.",
        "William Shakespeare wrote Hamlet.",
    ]

    predicted_answers = [
        "Paris is the capital of France.",
        "Hamlet was written by Shakespeare.",
    ]

    retrieved_contexts = [
        [
            "France is a country in Europe.",
            "Paris is the capital and largest city of France.",
        ],
        [
            "Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
        ],
    ]

    gold_contexts_list = [
        ["Paris is the capital and largest city of France."],
        ["Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601."],
    ]

    results = evaluate_rag_batch(
        questions=questions,
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        retrieved_contexts=retrieved_contexts,
        compute_semantic_metrics=True,
        bertscore_lang="en",
    )

    summary = summarize_rag_results(results)

    print("\nPer-example results:")
    for result in results:
        print(result)

    print("\nSummary:")
    print(summary)