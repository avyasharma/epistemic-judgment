"""
Epistemic Gating Module — LLM-as-a-Judge
=========================================
Evaluates retrieved passages along three dimensions before allowing generation:
  - Relevance:    How well do the passages match the query?
  - Sufficiency:  Do they contain enough information to answer the question?
  - Consistency:  Do they agree with each other (no contradictions)?

Two prompt modes (for ablation):
  - "structured": Per-dimension scores (0-10) + final decision + justification
  - "minimal":    Final decision (ANSWER / ABSTAIN) + one-line justification

Usage (library):
    judge = EpistemicJudge(client, mode="structured", threshold=6.0)
    gated_rag = GatedRAG(retriever, generator, judge)
    result = gated_rag.answer("What is X?")
    print(result)

CLI (see ``python epistemic_gated_rag.py --help``):
    Single run defaults to ``structured_<threshold>`` in the output filename.
    ``--ablations structured_5,structured_6,minimal`` runs all on the same
    retrieval; multiple structured thresholds share one judge LLM call per example.

Frontend / one-off query:
    ``--question "..."`` — retrieve, judge, generate; prints one JSON object (stdout or
    ``--serve-output``). Use ``--index-dir`` (built via ``python build_retriever_index.py``)
    so passages are FAISS-matched live with **query encoding only**, not a full re-index.
    ``--request-json path`` — JSON body ``{"question":"...", "retrieved_context": [...]}``;
    if ``retrieved_context`` is missing or empty, runs retrieval (``--index-dir`` or corpus).
    Import ``serve_epistemic_request`` for programmatic use without the batch CLI.
"""

# from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
from rag import (
    DocumentRetriever,
    OpenAIClient,
    VanillaRAG,
    default_squad_json_path,
    load_documents_for_retriever,
    load_squad_json_records,
)
from squad_utils import RagOutputsResponseCache, gold_answer_label_for_squad_record

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
BASE_URL = os.getenv("BASE_URL")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EpistemicVerdict:
    """Structured output from the epistemic judge."""
    decision: str                        # "ANSWER" or "ABSTAIN"
    justification: str

    # Populated in "structured" mode only
    relevance_score: Optional[float] = None
    sufficiency_score: Optional[float] = None
    consistency_score: Optional[float] = None
    composite_score: Optional[float] = None

    raw_response: str = ""               # always stored for debugging

    @property
    def should_answer(self) -> bool:
        return self.decision == "ANSWER"


@dataclass
class GatedRAGResult:
    """Full result returned by GatedRAG.answer()."""
    query: str
    verdict: EpistemicVerdict
    response: Optional[str]              # None when abstained
    retrieved_passages: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Query     : {self.query}",
            f"Decision  : {self.verdict.decision}",
            f"Why       : {self.verdict.justification}",
        ]
        if self.verdict.composite_score is not None:
            lines.append(
                f"Scores    : relevance={self.verdict.relevance_score:.1f}  "
                f"sufficiency={self.verdict.sufficiency_score:.1f}  "
                f"consistency={self.verdict.consistency_score:.1f}  "
                f"composite={self.verdict.composite_score:.1f}"
            )
        lines.append(f"Response  : {self.response or '[ABSTAINED]'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

STRUCTURED_SYSTEM = """\
You are an epistemic judge for a retrieval-augmented generation system.
Evaluate whether the retrieved passages provide sufficient, relevant, and consistent
evidence to answer the user query.

Score each dimension from 0 to 10:
  relevance   — 0=completely off-topic,          10=directly addresses the query
  sufficiency — 0=no useful information present,  10=fully enough to answer
  consistency — 0=passages heavily contradict,    10=fully coherent and agreeing

Then decide:
  ANSWER  — evidence is strong enough to generate a reliable answer
  ABSTAIN — evidence is too weak, irrelevant, or contradictory

Reply ONLY with a valid JSON object, no extra text:
{
  "relevance_score": <float 0-10>,
  "sufficiency_score": <float 0-10>,
  "consistency_score": <float 0-10>,
  "decision": "ANSWER" or "ABSTAIN",
  "justification": "<one or two sentences>"
}"""

STRUCTURED_USER = """\
Query: {query}

Retrieved Passages:
{passages}

Return the JSON verdict."""


MINIMAL_SYSTEM = """\
You are an epistemic judge for a retrieval-augmented generation system.
Decide whether the retrieved passages provide enough reliable evidence to answer
the user query.

Reply ONLY with a valid JSON object, no extra text:
{
  "decision": "ANSWER" or "ABSTAIN",
  "justification": "<one sentence>"
}"""

MINIMAL_USER = """\
Query: {query}

Retrieved Passages:
{passages}

Should the system answer or abstain?"""


PROMPT_TEMPLATES = {
    "structured": (STRUCTURED_SYSTEM, STRUCTURED_USER),
    "minimal":    (MINIMAL_SYSTEM,    MINIMAL_USER),
}


# ---------------------------------------------------------------------------
# EpistemicJudge
# ---------------------------------------------------------------------------

class EpistemicJudge:
    """
    LLM-as-a-judge epistemic gating module.

    Parameters
    ----------
    client : OpenAIClient
        LLM client — can be a smaller/cheaper model than the generator.
    mode : "structured" | "minimal"
        Prompt format to use; swap for ablation experiments.
    threshold : float
        Minimum weighted composite score to allow generation (structured mode only).
    weights : (float, float, float)
        Weights for (relevance, sufficiency, consistency). Must sum to 1.
    """

    def __init__(
        self,
        client: OpenAIClient,
        mode: str = "structured",
        threshold: float = 6.0,
        weights: tuple = (0.35, 0.45, 0.20),
    ):
        if mode not in PROMPT_TEMPLATES:
            raise ValueError(f"mode must be one of {list(PROMPT_TEMPLATES)!r}")
        self.client = client
        self.mode = mode
        self.threshold = threshold
        self.weights = weights

    def judge(self, query: str, passages: List[str]) -> EpistemicVerdict:
        """Evaluate passages for query and return a verdict."""
        system_prompt, user_template = PROMPT_TEMPLATES[self.mode]
        user_prompt = user_template.format(
            query=query,
            passages=self._format_passages(passages),
        )
        raw = self._call_llm(system_prompt, user_prompt)
        return self._parse(raw)

    def structured_judge_raw(self, query: str, passages: List[str]) -> str:
        """Single structured-prompt LLM call (scores only); threshold applied via :meth:`verdict_from_structured_raw`."""
        system_prompt, user_template = PROMPT_TEMPLATES["structured"]
        user_prompt = user_template.format(
            query=query,
            passages=self._format_passages(passages),
        )
        return self._call_llm(system_prompt, user_prompt)

    @staticmethod
    def verdict_from_structured_raw(
        raw: str,
        *,
        threshold: float,
        weights: tuple = (0.35, 0.45, 0.20),
    ) -> EpistemicVerdict:
        """
        Parse a structured judge response and apply ``threshold`` to the composite score.

        Used to sweep multiple thresholds without additional LLM calls.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[EpistemicJudge] parse error: {e}  raw={raw!r}")
            return EpistemicVerdict(
                decision="ABSTAIN",
                justification="Judge response unparseable; defaulting to abstain.",
                raw_response=raw,
            )

        justification = str(data.get("justification", ""))
        rel = float(data.get("relevance_score", 0))
        suf = float(data.get("sufficiency_score", 0))
        con = float(data.get("consistency_score", 0))
        w_r, w_s, w_c = weights
        composite = w_r * rel + w_s * suf + w_c * con
        decision = "ANSWER" if composite >= threshold else "ABSTAIN"
        return EpistemicVerdict(
            decision=decision,
            justification=justification,
            relevance_score=rel,
            sufficiency_score=suf,
            consistency_score=con,
            composite_score=round(composite, 2),
            raw_response=raw,
        )

    # ── private ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_passages(passages: List[str]) -> str:
        return "\n\n".join(
            f"[Passage {i+1}]: {p.strip()}" for i, p in enumerate(passages)
        )

    def _call_llm(self, system: str, user: str) -> str:
        resp = self.client.client.chat.completions.create(
            model=self.client.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    def _parse(self, raw: str) -> EpistemicVerdict:
        if self.mode == "structured":
            return self.verdict_from_structured_raw(
                raw, threshold=self.threshold, weights=self.weights
            )

        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[EpistemicJudge] parse error: {e}  raw={raw!r}")
            return EpistemicVerdict(
                decision="ABSTAIN",
                justification="Judge response unparseable; defaulting to abstain.",
                raw_response=raw,
            )

        decision = str(data.get("decision", "ABSTAIN")).upper()
        if decision not in ("ANSWER", "ABSTAIN"):
            decision = "ABSTAIN"
        justification = str(data.get("justification", ""))

        return EpistemicVerdict(
            decision=decision,
            justification=justification,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# GatedRAG
# ---------------------------------------------------------------------------

ABSTAIN_MESSAGE = (
    "I cannot provide a reliable answer because the retrieved evidence "
    "was deemed insufficient, irrelevant, or contradictory."
)


def default_rag_outputs_cache_path() -> str:
    """Path to vanilla RAG JSON for reuse of ``response`` (env or ``squad_rag_outputs.json``)."""
    return os.getenv("SQUAD_RAG_OUTPUTS_JSON", "squad_rag_outputs.json")


class GatedRAG:
    """
    Drop-in replacement for VanillaRAG with epistemic gating.

    Flow: retrieve → judge → generate (only if ANSWER), or judge → generate when
    using precomputed passages from ``answer_with_passages``.
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever],
        generator: OpenAIClient,
        judge: EpistemicJudge,
        abstain_message: str = ABSTAIN_MESSAGE,
        response_cache: Optional[RagOutputsResponseCache] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.judge = judge
        self.abstain_message = abstain_message
        self.response_cache = response_cache
        self._vanilla = (
            VanillaRAG(retriever, generator) if retriever is not None else None
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        *,
        example_key: Optional[str] = None,
        ids: Optional[Any] = None,
    ) -> GatedRAGResult:
        if self.retriever is None:
            raise TypeError(
                "GatedRAG has no retriever; use answer_with_passages(query, passages) "
                "with precomputed retrieval (see RETRIEVAL_JSON workflow)."
            )
        passages = self.retriever.retrieve(query, top_k=top_k)
        return self.answer_with_passages(
            query, passages, example_key=example_key, ids=ids
        )

    def answer_with_passages(
        self,
        query: str,
        passages: List[str],
        *,
        example_key: Optional[str] = None,
        ids: Optional[Any] = None,
    ) -> GatedRAGResult:
        verdict = self.judge.judge(query, passages)
        if verdict.should_answer:
            cached: Optional[str] = None
            if self.response_cache is not None:
                cached = self.response_cache.lookup(
                    example_key=example_key, ids=ids, question=query
                )
            response = (
                cached
                if cached is not None
                else self.generator.generate(query, passages)
            )
        else:
            response = None
        return GatedRAGResult(
            query=query,
            verdict=verdict,
            response=response,
            retrieved_passages=list(passages),
        )

    def answer_vanilla(self, query: str, top_k: int = 5):
        """Run without gating — for direct baseline comparison."""
        if self._vanilla is None:
            raise TypeError("answer_vanilla requires a retriever.")
        return self._vanilla.answer(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gated_pipeline(
    document_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    generator_model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    judge_mode: str = "structured",
    threshold: float = 6.0,
    weights: tuple = (0.35, 0.45, 0.20),
    rag_outputs_cache_path: Optional[str] = None,
) -> GatedRAG:
    """Build a fully wired GatedRAG pipeline."""
    from rag import load_documents_for_retriever

    print(f"[build_gated_pipeline] mode={judge_mode!r}  threshold={threshold}")

    retriever    = DocumentRetriever(model_name=embedding_model)
    generator    = OpenAIClient(model=generator_model)
    judge_client = OpenAIClient(model=judge_model)
    judge        = EpistemicJudge(
        judge_client, mode=judge_mode, threshold=threshold, weights=weights
    )

    docs, _ = load_documents_for_retriever(document_path)
    print(f"Loaded {len(docs)} raw documents")
    retriever.add_documents(docs)
    print(f"Indexed {len(retriever.documents)} chunks")

    cache_path = (
        rag_outputs_cache_path
        if rag_outputs_cache_path is not None
        else default_rag_outputs_cache_path()
    )
    response_cache = RagOutputsResponseCache.load(cache_path)
    if response_cache is not None:
        print(f"[build_gated_pipeline] Reusing answers from {cache_path!r} when judge ANSWERs")

    return GatedRAG(retriever, generator, judge, response_cache=response_cache)


def build_gated_judge_only_pipeline(
    generator_model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    judge_mode: str = "structured",
    threshold: float = 6.0,
    weights: tuple = (0.35, 0.45, 0.20),
    rag_outputs_cache_path: Optional[str] = None,
) -> GatedRAG:
    """
    Judge + generator without a retriever — use with precomputed ``RETRIEVAL_JSON``
    and :meth:`GatedRAG.answer_with_passages`.
    """
    print(f"[build_gated_judge_only_pipeline] mode={judge_mode!r}  threshold={threshold}")
    generator = OpenAIClient(model=generator_model)
    judge_client = OpenAIClient(model=judge_model)
    judge = EpistemicJudge(
        judge_client, mode=judge_mode, threshold=threshold, weights=weights
    )
    cache_path = (
        rag_outputs_cache_path
        if rag_outputs_cache_path is not None
        else default_rag_outputs_cache_path()
    )
    response_cache = RagOutputsResponseCache.load(cache_path)
    if response_cache is not None:
        print(
            f"[build_gated_judge_only_pipeline] Reusing answers from {cache_path!r} when judge ANSWERs"
        )
    return GatedRAG(
        retriever=None, generator=generator, judge=judge, response_cache=response_cache
    )


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    pipeline_kwargs: dict,
    questions: List[str],
    gold_answers: List[str],
    configs: List[dict],
    output_path: str = "ablation_outputs.json",
    top_k: int = 5,
) -> dict:
    """
    Run multiple judge configs over the same question set.

    Indexes the corpus once and retrieves once per question. Several
    ``structured`` thresholds share a single structured judge LLM call per
    question; ``minimal`` uses an extra call when present.

    Parameters
    ----------
    pipeline_kwargs : dict
        Passed to ``build_gated_pipeline`` for the first config merge only
        (document path, models, cache path, etc.). Per-config entries override
        keys in later processing where noted.
    configs : list[dict]
        Each entry needs a ``name`` key plus optional ``judge_mode``,
        ``threshold``, ``weights``, and other keys merged into the first
        pipeline build.

        Example:
            [
                {"name": "structured_t5",  "judge_mode": "structured", "threshold": 5.0},
                {"name": "structured_t6",  "judge_mode": "structured", "threshold": 6.0},
                {"name": "structured_t7",  "judge_mode": "structured", "threshold": 7.0},
                {"name": "minimal",        "judge_mode": "minimal"},
            ]
    """
    rows: List[Tuple[str, dict]] = []
    for cfg in configs:
        c = dict(cfg)
        name = c.pop("name")
        rows.append((name, c))

    merged0 = {**pipeline_kwargs, **rows[0][1]}
    print(f"\n{'='*60}\n[run_ablation] building shared index (first config merge)\n{'='*60}")
    p0 = build_gated_pipeline(**merged0)
    retriever = p0.retriever
    generator = p0.generator
    response_cache = p0.response_cache
    judge_client = p0.judge.client
    default_weights = tuple(p0.judge.weights)

    needs_structured = any(
        str(extra.get("judge_mode", "structured")).lower() == "structured"
        for _, extra in rows
    )
    needs_minimal = any(
        str(extra.get("judge_mode", "structured")).lower() == "minimal"
        for _, extra in rows
    )
    structured_caller = (
        EpistemicJudge(
            judge_client, mode="structured", threshold=6.0, weights=default_weights
        )
        if needs_structured
        else None
    )
    minimal_judge = (
        EpistemicJudge(
            judge_client, mode="minimal", threshold=0.0, weights=default_weights
        )
        if needs_minimal
        else None
    )

    all_outputs: dict = {name: [] for name, _ in rows}
    nq = len(questions)

    for i, (q, ans) in enumerate(zip(questions, gold_answers)):
        passages = retriever.retrieve(q, top_k=top_k)
        structured_raw: Optional[str] = None
        if needs_structured:
            assert structured_caller is not None
            structured_raw = structured_caller.structured_judge_raw(q, passages)

        for (name, extra) in rows:
            mode = str(extra.get("judge_mode", "structured")).lower()
            w = tuple(extra.get("weights", default_weights))
            if len(w) != 3:
                w = default_weights
            if mode == "structured":
                th = float(extra.get("threshold", 6.0))
                assert structured_raw is not None
                verdict = EpistemicJudge.verdict_from_structured_raw(
                    structured_raw, threshold=th, weights=w
                )
            else:
                assert minimal_judge is not None
                verdict = minimal_judge.judge(q, passages)

            response = _response_for_verdict(
                verdict,
                q,
                passages,
                generator,
                response_cache,
                example_key=str(i),
                ids=None,
            )
            all_outputs[name].append(
                {
                    "question": q,
                    "gold_answer": ans,
                    "decision": verdict.decision,
                    "justification": verdict.justification,
                    "relevance_score": verdict.relevance_score,
                    "sufficiency_score": verdict.sufficiency_score,
                    "consistency_score": verdict.consistency_score,
                    "composite_score": verdict.composite_score,
                    "response": response,
                    "retrieved_passages": list(passages),
                }
            )

        if (i + 1) % 500 == 0 or (i + 1) == nq:
            print(f"  [{i + 1}/{nq}] (all configs)")

    for name in all_outputs:
        _summary(name, all_outputs[name])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"\nAblation results saved to {output_path}")
    return all_outputs


def _summary(name: str, results: list) -> None:
    n = len(results)
    answered = sum(1 for r in results if r["decision"] == "ANSWER")
    print(f"\n--- {name} ---  answered={answered}/{n}  abstained={n-answered}/{n}")




def fill_abstain_responses(data: dict) -> dict:
    for key, entry in data.items():
        if entry.get("decision", "").upper() == "ABSTAIN":
            entry["response"] = "I do not know."
    return data


# ---------------------------------------------------------------------------
# main — drop-in replacement for rag.py main
# ---------------------------------------------------------------------------

def _sort_inference_items(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    def key_fn(item: Tuple[str, Any]) -> Tuple[int, Any]:
        k = item[0]
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(data.items(), key=key_fn)


def _gold_from_retrieval_ex(ex: Dict[str, Any]) -> str:
    for key in ("gold_answer", "gold answer"):
        v = ex.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    alts = ex.get("gold_answers")
    if isinstance(alts, list) and alts:
        return str(alts[0]).strip()
    return ""


def _passages_from_retrieval_ex(ex: Dict[str, Any]) -> List[str]:
    ctx = ex.get("retrieved_context")
    if ctx is None:
        ctx = ex.get("retrieved_passages")
    if isinstance(ctx, str):
        return [ctx] if ctx.strip() else []
    if isinstance(ctx, list):
        return [str(p) for p in ctx if p is not None and str(p).strip()]
    return []


@dataclass(frozen=True)
class AblationSpec:
    """CLI / batch identifier, e.g. ``structured_6`` or ``minimal``."""

    name: str
    mode: str
    threshold: Optional[float] = None


def parse_ablation_token(token: str) -> AblationSpec:
    low = token.strip().lower()
    if not low:
        raise ValueError("empty ablation token in --ablations")
    if low == "minimal":
        return AblationSpec(name="minimal", mode="minimal", threshold=None)
    if low.startswith("structured_"):
        rest = low[len("structured_") :]
        if not rest:
            raise ValueError(
                f"invalid ablation {token!r}: use structured_<threshold>, e.g. structured_6"
            )
        thresh = float(rest)
        disp = str(int(thresh)) if thresh == int(thresh) else str(thresh)
        return AblationSpec(name=f"structured_{disp}", mode="structured", threshold=thresh)
    raise ValueError(
        f"unknown ablation {token!r}; use minimal or structured_<number> (e.g. structured_5)"
    )


def spec_from_mode_threshold(mode: str, threshold: float) -> AblationSpec:
    m = mode.strip().lower()
    if m == "minimal":
        return AblationSpec(name="minimal", mode="minimal", threshold=None)
    if m == "structured":
        th = float(threshold)
        disp = str(int(th)) if th == int(th) else str(th)
        return AblationSpec(name=f"structured_{disp}", mode="structured", threshold=th)
    raise ValueError(f"judge mode must be structured or minimal, got {mode!r}")


def resolve_ablation_specs(
    ablations_csv: Optional[str], judge_mode: str, threshold: float
) -> List[AblationSpec]:
    if ablations_csv and ablations_csv.strip():
        parts = [p.strip() for p in ablations_csv.split(",") if p.strip()]
        return [parse_ablation_token(p) for p in parts]
    return [spec_from_mode_threshold(judge_mode, threshold)]


def _response_for_verdict(
    verdict: EpistemicVerdict,
    query: str,
    passages: List[str],
    generator: OpenAIClient,
    response_cache: Optional[RagOutputsResponseCache],
    *,
    example_key: Optional[str],
    ids: Optional[Any],
) -> Optional[str]:
    if not verdict.should_answer:
        return None
    if response_cache is not None:
        cached = response_cache.lookup(
            example_key=example_key, ids=ids, question=query
        )
        if cached is not None:
            return cached
    return generator.generate(query, passages)


def _record_retrieval_row(
    _k: str,
    ex: Dict[str, Any],
    q: str,
    passages: List[str],
    verdict: EpistemicVerdict,
    response: Optional[str],
) -> Dict[str, Any]:
    gold = _gold_from_retrieval_ex(ex)
    return {
        "question": q,
        "gold_answer": gold,
        "gold_answers": ex.get("gold_answers") or [],
        "is_impossible": ex.get("is_impossible", False),
        "ids": ex.get("ids"),
        "decision": verdict.decision,
        "justification": verdict.justification,
        "relevance_score": verdict.relevance_score,
        "sufficiency_score": verdict.sufficiency_score,
        "consistency_score": verdict.consistency_score,
        "composite_score": verdict.composite_score,
        "response": response,
        "retrieved_context": list(passages),
    }


def _record_squad_row(
    rec: Dict[str, Any],
    idx: str,
    verdict: EpistemicVerdict,
    response: Optional[str],
    passages: List[str],
) -> Dict[str, Any]:
    gold = gold_answer_label_for_squad_record(rec)
    return {
        "question": rec["question"],
        "gold_answer": gold,
        "gold_answers": rec.get("answers") or [],
        "is_impossible": rec.get("is_impossible", False),
        "ids": rec.get("ids"),
        "decision": verdict.decision,
        "justification": verdict.justification,
        "relevance_score": verdict.relevance_score,
        "sufficiency_score": verdict.sufficiency_score,
        "consistency_score": verdict.consistency_score,
        "composite_score": verdict.composite_score,
        "response": response,
        "retrieved_context": list(passages),
    }


def epistemic_response_json(
    query: str,
    passages: List[str],
    verdict: EpistemicVerdict,
    response: Optional[str],
    *,
    gold_answer: Optional[str] = None,
    gold_answers: Optional[List[Any]] = None,
    is_impossible: Optional[bool] = None,
    ids: Optional[Any] = None,
) -> Dict[str, Any]:
    """Serializable row matching batch gated JSON (for API / frontend)."""
    row: Dict[str, Any] = {
        "question": query,
        "decision": verdict.decision,
        "justification": verdict.justification,
        "relevance_score": verdict.relevance_score,
        "sufficiency_score": verdict.sufficiency_score,
        "consistency_score": verdict.consistency_score,
        "composite_score": verdict.composite_score,
        "response": response,
        "retrieved_context": list(passages),
    }
    if gold_answer is not None:
        row["gold_answer"] = gold_answer
    if gold_answers is not None:
        row["gold_answers"] = gold_answers
    if is_impossible is not None:
        row["is_impossible"] = is_impossible
    if ids is not None:
        row["ids"] = ids
    return row


def _verdicts_for_ablation_specs(
    q: str,
    passages: List[str],
    specs: List[AblationSpec],
    weights: Tuple[float, float, float],
    structured_caller: Optional[EpistemicJudge],
    minimal_judge: Optional[EpistemicJudge],
) -> Dict[str, EpistemicVerdict]:
    needs_structured = any(s.mode == "structured" for s in specs)
    structured_raw: Optional[str] = None
    if needs_structured:
        assert structured_caller is not None
        structured_raw = structured_caller.structured_judge_raw(q, passages)
    out: Dict[str, EpistemicVerdict] = {}
    for spec in specs:
        if spec.mode == "structured":
            assert structured_raw is not None and spec.threshold is not None
            out[spec.name] = EpistemicJudge.verdict_from_structured_raw(
                structured_raw,
                threshold=spec.threshold,
                weights=weights,
            )
        else:
            assert minimal_judge is not None
            out[spec.name] = minimal_judge.judge(q, passages)
    return out


def serve_epistemic_request(
    *,
    question: str,
    retrieved_context: Optional[List[str]] = None,
    squad_json_path: Optional[str] = None,
    index_dir: Optional[str] = None,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    ablations: Optional[str] = None,
    judge_mode: str = "structured",
    threshold: float = 6.0,
    generator_model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    rag_outputs_cache_path: Optional[str] = None,
    example_key: Optional[str] = None,
    ids: Optional[Any] = None,
    gold_answer: Optional[str] = None,
    gold_answers: Optional[List[Any]] = None,
    is_impossible: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run epistemic judge (+ retrieval if needed) and optional generation once.

    If ``retrieved_context`` is non-empty, uses those passages (no retrieval).

    Otherwise, if ``index_dir`` is set, loads a pre-built FAISS index from disk
    (:meth:`rag.DocumentRetriever.save_index` / ``build_retriever_index.py``) and
    only encodes the question at request time.

    Else falls back to indexing ``squad_json_path`` (or ``default_squad_json_path()``)
    on every call (slow for interactive use).

    Returns a single example dict if one ablation spec, else
    ``{"ablations": {name: dict, ...}}``. When several specs all ``ANSWER``,
    generation runs at most once and the same ``response`` is attached to each.
    """
    q = str(question).strip()
    if not q:
        raise ValueError("question must be non-empty")

    specs = resolve_ablation_specs(ablations, judge_mode, threshold)
    weights: Tuple[float, float, float] = (0.35, 0.45, 0.20)

    passages = list(retrieved_context or [])
    passages = [str(p) for p in passages if p is not None and str(p).strip()]
    if not passages:
        idx = str(index_dir).strip() if index_dir else ""
        if idx:
            if not os.path.isdir(idx):
                raise FileNotFoundError(
                    f"index_dir {idx!r} is not a directory. Build it: python build_retriever_index.py …"
                )
            retriever = DocumentRetriever.load_index(idx)
            passages = retriever.retrieve(q, top_k=top_k)
        else:
            corpus = squad_json_path or default_squad_json_path()
            if not os.path.isfile(corpus):
                raise FileNotFoundError(
                    f"No passages provided and no index_dir; SQuAD corpus not found at {corpus!r}. "
                    "Pass retrieved_context, index_dir, or squad_json_path."
                )
            retriever = DocumentRetriever(model_name=embedding_model)
            docs, _ = load_documents_for_retriever(corpus)
            retriever.add_documents(docs)
            passages = retriever.retrieve(q, top_k=top_k)

    cache = (
        RagOutputsResponseCache.load(rag_outputs_cache_path)
        if rag_outputs_cache_path
        else None
    )
    generator = OpenAIClient(model=generator_model)
    judge_client = OpenAIClient(model=judge_model)
    needs_structured = any(s.mode == "structured" for s in specs)
    needs_minimal = any(s.mode == "minimal" for s in specs)
    structured_caller = (
        EpistemicJudge(judge_client, mode="structured", threshold=6.0, weights=weights)
        if needs_structured
        else None
    )
    minimal_judge = (
        EpistemicJudge(judge_client, mode="minimal", threshold=0.0, weights=weights)
        if needs_minimal
        else None
    )

    verdicts = _verdicts_for_ablation_specs(
        q, passages, specs, weights, structured_caller, minimal_judge
    )

    shared_response: Optional[str] = None
    first_answer_verdict: Optional[EpistemicVerdict] = None
    for v in verdicts.values():
        if v.should_answer:
            first_answer_verdict = v
            break
    if first_answer_verdict is not None:
        shared_response = _response_for_verdict(
            first_answer_verdict,
            q,
            passages,
            generator,
            cache,
            example_key=example_key,
            ids=ids,
        )

    def row_for(name: str) -> Dict[str, Any]:
        v = verdicts[name]
        resp = shared_response if v.should_answer else None
        return epistemic_response_json(
            q,
            passages,
            v,
            resp,
            gold_answer=gold_answer,
            gold_answers=gold_answers,
            is_impossible=is_impossible,
            ids=ids,
        )

    if len(specs) == 1:
        return row_for(specs[0].name)
    return {"ablations": {s.name: row_for(s.name) for s in specs}}


def _load_json_request_file(path: str) -> Dict[str, Any]:
    if path.strip() == "-":
        data = json.load(sys.stdin)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Request JSON must be an object, e.g. {\"question\": \"...\"}")
    return data


def _run_serve_mode(args: argparse.Namespace) -> None:
    if args.question is not None and args.request_json is not None:
        raise SystemExit("Use either --question or --request-json, not both.")
    if args.question is not None:
        payload: Dict[str, Any] = {"question": args.question}
    else:
        assert args.request_json is not None
        payload = _load_json_request_file(args.request_json)

    q = str(payload.get("question", "")).strip()
    if not q:
        raise SystemExit("Missing or empty \"question\" in request.")

    passages_in = _passages_from_retrieval_ex(payload)
    ga = _gold_from_retrieval_ex(payload)
    ganswers = payload.get("gold_answers")
    idx_dir = args.index_dir
    if not idx_dir and isinstance(payload.get("index_dir"), str):
        idx_dir = payload["index_dir"].strip() or None
    out = serve_epistemic_request(
        question=q,
        retrieved_context=passages_in if passages_in else None,
        squad_json_path=args.squad_json,
        index_dir=idx_dir,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        ablations=args.ablations,
        judge_mode=args.judge_mode,
        threshold=args.threshold,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        rag_outputs_cache_path=args.rag_cache
        if os.path.isfile(args.rag_cache)
        else None,
        example_key=str(payload.get("example_key", "")).strip() or None,
        ids=payload.get("ids"),
        gold_answer=ga if ga.strip() else None,
        gold_answers=ganswers if isinstance(ganswers, list) and ganswers else None,
        is_impossible=payload.get("is_impossible")
        if "is_impossible" in payload
        else None,
    )

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.serve_output:
        with open(args.serve_output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote JSON to {args.serve_output!r}")
    else:
        print(text)


def _write_ablation_outputs(
    nested: Dict[str, Dict[str, Any]],
    specs: List[AblationSpec],
    *,
    output: Optional[str],
    output_prefix: str,
    split_outputs: bool,
) -> None:
    if split_outputs:
        for s in specs:
            path = f"{output_prefix}_{s.name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nested[s.name], f, ensure_ascii=False, indent=2)
            print(f"Results saved to {path}")
        return

    if len(specs) == 1:
        path = (
            output
            or os.getenv("GATED_OUTPUTS_JSON")
            or f"{output_prefix}_{specs[0].name}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nested[specs[0].name], f, ensure_ascii=False, indent=2)
        print(f"Results saved to {path}")
        return

    path = output or "squad_gated_ablations.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Epistemic gated RAG: run over squad_retrieval.json (judge+generate) "
            "or index SQuAD JSON when no retrieval file exists."
        )
    )
    p.add_argument(
        "--retrieval-json",
        default=os.getenv("RETRIEVAL_JSON", "squad_retrieval.json"),
        help="Precomputed retrieval JSON object (skips embedding index when present).",
    )
    p.add_argument(
        "--squad-json",
        default=None,
        help="SQuAD JSON array path when retrieval file is missing (default: rag.default_squad_json_path()).",
    )
    p.add_argument(
        "--rag-cache",
        default=os.getenv("SQUAD_RAG_OUTPUTS_JSON", "squad_rag_outputs.json"),
        help="Vanilla RAG outputs JSON for answer reuse when the judge says ANSWER.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path: single ablation, or combined ablations JSON when multiple specs.",
    )
    p.add_argument(
        "--output-prefix",
        default="squad_gated_outputs",
        help="Filename prefix for defaults and for --split-outputs (e.g. squad_gated_outputs_structured_6.json).",
    )
    p.add_argument(
        "--split-outputs",
        action="store_true",
        help="With multiple --ablations, write one file per spec instead of one nested JSON.",
    )
    p.add_argument(
        "--ablations",
        default=None,
        help=(
            "Comma-separated ablations: structured_<threshold> (e.g. structured_5,structured_7) "
            "and/or minimal. Multiple structured_* thresholds share one judge LLM call per example."
        ),
    )
    p.add_argument(
        "--judge-mode",
        default=os.getenv("JUDGE_MODE", "structured"),
        help="Used when --ablations is omitted: structured | minimal.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("JUDGE_THRESHOLD", "6.0")),
        help="Used when --ablations is omitted and judge-mode is structured.",
    )
    p.add_argument("--generator-model", default="gpt-5-mini")
    p.add_argument("--judge-model", default="gpt-5-mini")
    p.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Only used when indexing SQuAD (no retrieval JSON).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retriever top-k when indexing SQuAD.",
    )
    p.add_argument(
        "--question",
        default=None,
        help=(
            "Frontend / one-off: run retrieve → epistemic judge → generate for this "
            "question; print JSON (or use --serve-output). Uses --squad-json / default corpus."
        ),
    )
    p.add_argument(
        "--request-json",
        default=None,
        metavar="PATH",
        help=(
            "Frontend / test JSON: object with \"question\" and optional "
            "\"retrieved_context\" / \"retrieved_passages\". If passages absent or empty, "
            "runs full retrieval like --question. Use path \"-\" to read stdin."
        ),
    )
    p.add_argument(
        "--serve-output",
        default=None,
        metavar="PATH",
        help="With --question or --request-json, write JSON here instead of printing.",
    )
    p.add_argument(
        "--index-dir",
        default=None,
        metavar="DIR",
        help=(
            "Pre-built FAISS index directory (from ``python build_retriever_index.py …``). "
            "Serve / live queries only encode the question + search."
        ),
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.question is not None or args.request_json is not None:
        _run_serve_mode(args)
        return

    specs = resolve_ablation_specs(args.ablations, args.judge_mode, args.threshold)
    weights: Tuple[float, float, float] = (0.35, 0.45, 0.20)

    rag_cache_path = args.rag_cache
    response_cache = RagOutputsResponseCache.load(rag_cache_path)
    if response_cache is not None:
        print(f"[main] Reusing answers from {rag_cache_path!r} when judge ANSWERs")

    generator = OpenAIClient(model=args.generator_model)
    judge_client = OpenAIClient(model=args.judge_model)
    needs_structured = any(s.mode == "structured" for s in specs)
    needs_minimal = any(s.mode == "minimal" for s in specs)

    structured_caller = (
        EpistemicJudge(judge_client, mode="structured", threshold=6.0, weights=weights)
        if needs_structured
        else None
    )
    minimal_judge = (
        EpistemicJudge(judge_client, mode="minimal", threshold=0.0, weights=weights)
        if needs_minimal
        else None
    )

    nested: Dict[str, Dict[str, Any]] = {s.name: {} for s in specs}
    retrieval_path = args.retrieval_json

    if os.path.isfile(retrieval_path):
        with open(retrieval_path, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
        if not isinstance(retrieval_data, dict):
            raise ValueError(
                f"retrieval JSON must be a JSON object, got {type(retrieval_data).__name__}"
            )

        items = [
            (k, ex)
            for k, ex in _sort_inference_items(retrieval_data)
            if isinstance(ex, dict)
        ]
        n_items = len(items)
        for j, (k, ex) in enumerate(items, start=1):
            q = str(ex.get("question", ""))
            passages = _passages_from_retrieval_ex(ex)
            structured_raw: Optional[str] = None
            if needs_structured:
                assert structured_caller is not None
                structured_raw = structured_caller.structured_judge_raw(q, passages)

            for spec in specs:
                if spec.mode == "structured":
                    assert structured_raw is not None and spec.threshold is not None
                    verdict = EpistemicJudge.verdict_from_structured_raw(
                        structured_raw,
                        threshold=spec.threshold,
                        weights=weights,
                    )
                else:
                    assert minimal_judge is not None
                    verdict = minimal_judge.judge(q, passages)

                response = _response_for_verdict(
                    verdict,
                    q,
                    passages,
                    generator,
                    response_cache,
                    example_key=k,
                    ids=ex.get("ids"),
                )
                nested[spec.name][k] = _record_retrieval_row(
                    k, ex, q, passages, verdict, response
                )

            if j % 500 == 0 or j == n_items:
                print(f"Gated {j}/{n_items} (ablations: {', '.join(s.name for s in specs)})")

        _write_ablation_outputs(
            nested,
            specs,
            output=args.output,
            output_prefix=args.output_prefix,
            split_outputs=args.split_outputs,
        )
        return

    json_path = args.squad_json or default_squad_json_path()
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"No retrieval file at {retrieval_path!r} and no SQuAD file at {json_path!r}. "
            "Run rag.py to write squad_retrieval.json, or pass --retrieval-json / --squad-json."
        )

    records = load_squad_json_records(json_path)
    retriever = DocumentRetriever(model_name=args.embedding_model)
    docs, _ = load_documents_for_retriever(json_path)
    retriever.add_documents(docs)
    print(f"[main] Indexed {len(retriever.documents)} chunks from {json_path!r}")

    n_rec = len(records)
    for i, rec in enumerate(records):
        q = rec["question"]
        passages = retriever.retrieve(q, top_k=args.top_k)
        structured_raw = None
        if needs_structured:
            assert structured_caller is not None
            structured_raw = structured_caller.structured_judge_raw(q, passages)

        for spec in specs:
            if spec.mode == "structured":
                assert structured_raw is not None and spec.threshold is not None
                verdict = EpistemicJudge.verdict_from_structured_raw(
                    structured_raw,
                    threshold=spec.threshold,
                    weights=weights,
                )
            else:
                assert minimal_judge is not None
                verdict = minimal_judge.judge(q, passages)

            response = _response_for_verdict(
                verdict,
                q,
                passages,
                generator,
                response_cache,
                example_key=str(i),
                ids=rec.get("ids"),
            )
            nested[spec.name][str(i)] = _record_squad_row(
                rec, str(i), verdict, response, passages
            )

        if (i + 1) % 500 == 0 or (i + 1) == n_rec:
            print(
                f"Done with Question {i + 1}/{n_rec} (ablations: {', '.join(s.name for s in specs)})"
            )

    _write_ablation_outputs(
        nested,
        specs,
        output=args.output,
        output_prefix=args.output_prefix,
        split_outputs=args.split_outputs,
    )


if __name__ == "__main__":
    main()