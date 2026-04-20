"""
SQuAD file utilities (sampling, etc.) kept separate from ``rag.py`` to avoid
pulling retrieval / embedding dependencies when only JSON manipulation is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _sort_str_dict_items(data: Dict[str, Any]) -> List[Tuple[str, Any]]:
    def key_fn(item: Tuple[str, Any]) -> Tuple[int, Any]:
        k = item[0]
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(data.items(), key=key_fn)


@dataclass
class RagOutputsResponseCache:
    """
    Map from vanilla RAG outputs JSON (string-keyed dict) to ``response`` text.

    Lookup order: same top-level example key → ``ids`` → exact ``question``.
    """

    by_example_key: Dict[str, str] = field(default_factory=dict)
    by_id: Dict[str, str] = field(default_factory=dict)
    by_question: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def load(path: Optional[str]) -> Optional["RagOutputsResponseCache"]:
        if not path or not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        by_key: Dict[str, str] = {}
        by_id: Dict[str, str] = {}
        by_q: Dict[str, str] = {}
        for k, ex in _sort_str_dict_items(data):
            if not isinstance(ex, dict):
                continue
            resp = ex.get("response")
            if resp is None or not str(resp).strip():
                continue
            text = str(resp).strip()
            by_key[str(k)] = text
            oid = ex.get("ids")
            if oid is not None and str(oid).strip():
                by_id.setdefault(str(oid).strip(), text)
            q = str(ex.get("question", "")).strip()
            if q:
                by_q.setdefault(q, text)
        return RagOutputsResponseCache(by_example_key=by_key, by_id=by_id, by_question=by_q)

    def lookup(
        self,
        *,
        example_key: Optional[str] = None,
        ids: Optional[Any] = None,
        question: str = "",
    ) -> Optional[str]:
        if example_key is not None:
            hit = self.by_example_key.get(str(example_key))
            if hit:
                return hit
        if ids is not None and str(ids).strip():
            hit = self.by_id.get(str(ids).strip())
            if hit:
                return hit
        q = question.strip()
        if q:
            return self.by_question.get(q)
        return None


def ordered_gold_answers_shortest_first(answers: Any) -> List[str]:
    """
    Deduplicate SQuAD ``answers`` and order by increasing length (then lexicographic tie-break).

    SQuAD often lists a long span first and shorter aliases (e.g. \"GTE\" vs a full sentence);
    putting the shortest span first aligns gold labels and eval with extractive QA style.
    """
    if not isinstance(answers, list):
        return []
    raw = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not raw:
        return []
    unique: List[str] = list(dict.fromkeys(raw))
    return sorted(unique, key=lambda s: (len(s), s))


def gold_answer_label_for_squad_record(rec: Dict[str, Any]) -> str:
    """Primary gold string: shortest accepted answer, or unanswerable placeholder."""
    if rec.get("is_impossible"):
        return "I do not know."
    ordered = ordered_gold_answers_shortest_first(rec.get("answers") or [])
    return ordered[0] if ordered else ""


def load_squad_json_records(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {json_path}")
    return data


def sample_squad_records(
    records: List[Dict[str, Any]],
    fraction: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Return a random subset of ``records`` (without replacement), stable for ``seed``."""
    if not records:
        return []
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    rng = random.Random(seed)
    n = max(1, int(round(len(records) * fraction)))
    n = min(n, len(records))
    return rng.sample(records, n)


def write_sampled_squad_json(
    source_path: str,
    dest_path: str,
    *,
    fraction: float = 0.2,
    seed: int = 42,
) -> int:
    """Write a random ``fraction`` of SQuAD records from ``source_path`` to ``dest_path``."""
    records = load_squad_json_records(source_path)
    sampled = sample_squad_records(records, fraction, seed=seed)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    return len(sampled)


def main() -> None:
    parser = argparse.ArgumentParser(description="SQuAD JSON utilities.")
    parser.add_argument(
        "--sample-test",
        action="store_true",
        help="Write a random sample of a SQuAD array (default: 20%% of squad_test.json → squad_test_sampled.json).",
    )
    parser.add_argument("--source", default="squad_test.json", help="Input SQuAD JSON array.")
    parser.add_argument("--out", default="squad_test_sampled.json", help="Output JSON path.")
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction of rows to keep.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    if args.sample_test:
        if not os.path.isfile(args.source):
            raise FileNotFoundError(f"{args.source!r} not found.")
        n = write_sampled_squad_json(
            args.source,
            args.out,
            fraction=args.fraction,
            seed=args.seed,
        )
        print(f"Wrote {n} examples ({args.fraction:.0%}) to {args.out}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
