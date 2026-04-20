"""
Lightweight abstention model from query + retrieved passages only.

Uses BM25 (Okapi), passage/query length stats, query–passage lexical overlap,
softmax entropy over passage BM25 scores, optional dense (MiniLM) cosine
scores, query–passage lexical precision (no gold labels), and logistic
regression on a fixed feature vector (see ``_N_FEATURES``). Training is
numpy-only (gradient descent); no sklearn.

Gold-answer overlap precision is intentionally omitted: it would be unavailable
at inference or would leak labels when gold is only used for training.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Feature count for saved weights (change breaks old JSON checkpoints).
_N_FEATURES = 19

# --- tokenization ---

_STOP = frozenset(
    "a an the to of in for on at by from with without into over after before "
    "is are was were be been being it its this that these those as or and but "
    "if than then so such what which who whom whose how when where why can could "
    "should would may might must do does did has have had not no yes we you he she "
    "they them their our your my his her any some all each every both few more most "
    "other another one two about into through during including against between under "
    "again further once here there up down out off also only own same than too very "
    "just".split()
)


def tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", str(text).lower())


def content_tokens(text: Optional[str]) -> List[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) > 1]


# --- BM25 ---

def _bm25_scores(query_tokens: List[str], doc_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """BM25 score of query against each document (passage)."""
    n_docs = len(doc_tokens)
    if n_docs == 0 or not query_tokens:
        return []

    dl = np.array([max(1, len(d)) for d in doc_tokens], dtype=np.float64)
    avgdl = float(dl.mean())

    df: Dict[str, int] = {}
    for d in doc_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t in set(query_tokens):
        f = df.get(t, 0)
        idf[t] = math.log((n_docs - f + 0.5) / (f + 0.5) + 1.0)

    scores: List[float] = []
    for d in doc_tokens:
        tf: Dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        dl_i = max(1, len(d))
        s = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            f = tf[t]
            denom = f + k1 * (1.0 - b + b * dl_i / avgdl)
            s += idf[t] * (f * (k1 + 1.0)) / denom
        scores.append(s)
    return scores


def _softmax_entropy(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    a = np.array(xs, dtype=np.float64)
    a = a - np.max(a)
    ex = np.exp(np.clip(a, -50, 50))
    z = ex.sum()
    if z <= 0:
        return 0.0
    p = ex / z
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _max_jaccard(q: List[str], doc_tokens: List[List[str]]) -> float:
    if not q:
        return 0.0
    sq = set(q)
    best = 0.0
    for d in doc_tokens:
        sd = set(d)
        if not sd:
            continue
        inter = len(sq & sd)
        union = len(sq | sd)
        if union:
            best = max(best, inter / union)
    return best


def _query_passage_token_precisions(
    q_content: List[str], doc_tokens: List[List[str]]
) -> Tuple[float, float]:
    """
    Per-passage lexical precision: share of content query tokens (with multiplicity)
    that appear in each passage's token bag. No gold labels — safe at inference.

    (Gold-overlap ``precision`` would be label leakage when gold is not available.)
    """
    if not q_content or not doc_tokens:
        return 0.0, 0.0
    den = float(len(q_content))
    per_pass: List[float] = []
    for dt in doc_tokens:
        st = set(dt)
        hit = sum(1 for t in q_content if t in st)
        per_pass.append(hit / den)
    arr = np.array(per_pass, dtype=np.float64)
    return float(arr.max()), float(arr.mean())


def _dense_cosine_stats(
    question: str, passages: List[str], dense_encoder: Optional["DensePassageEncoder"]
) -> Tuple[float, float, float]:
    """Max / mean / softmax-entropy of cosine(query, passage) when encoder set; else zeros."""
    if dense_encoder is None or not passages or not str(question).strip():
        return 0.0, 0.0, 0.0
    cos = dense_encoder.cosine_scores(question, passages)
    if cos.size == 0:
        return 0.0, 0.0, 0.0
    return float(cos.max()), float(cos.mean()), _softmax_entropy(cos.tolist())


def extract_features(
    question: str,
    passages: Sequence[str],
    *,
    dense_encoder: Optional["DensePassageEncoder"] = None,
) -> np.ndarray:
    """
    Fixed-order feature vector (used for both training and inference).

    Order (19): BM25 max/mean/std/gap/entropy; log1p(n_pass); mean passage tokens;
    log1p(total tokens); content-token union coverage; max Jaccard (content);
    mean pairwise passage Jaccard; log1p(raw query tokens); min/median BM25;
    max/mean per-passage query-token precision; dense cosine max/mean/entropy.

    Dense block is zero when ``dense_encoder`` is None (must match training).
    """
    passages = [str(p) for p in passages if p is not None and str(p).strip()]
    q_raw = tokenize(question)
    q = content_tokens(question)
    doc_tokens = [tokenize(p) for p in passages]

    if not passages:
        return np.zeros(_N_FEATURES, dtype=np.float64)

    bm25 = _bm25_scores(q_raw, doc_tokens)
    bm25_np = np.array(bm25, dtype=np.float64)
    max_b = float(bm25_np.max()) if len(bm25_np) else 0.0
    mean_b = float(bm25_np.mean()) if len(bm25_np) else 0.0
    std_b = float(bm25_np.std()) if len(bm25_np) > 1 else 0.0
    sorted_b = np.sort(bm25_np)
    top2 = sorted_b[-2] if len(sorted_b) >= 2 else sorted_b[-1]
    gap = float(sorted_b[-1] - top2) if len(sorted_b) else 0.0
    ent = _softmax_entropy(bm25)
    min_b = float(bm25_np.min()) if len(bm25_np) else 0.0
    med_b = float(np.median(bm25_np)) if len(bm25_np) else 0.0

    lens = np.array([len(dt) for dt in doc_tokens], dtype=np.float64)
    mean_pw = float(lens.mean()) if len(lens) else 0.0
    total_pw = float(lens.sum())
    n_pass = float(len(passages))

    cov = 0.0
    if q:
        union_doc = set()
        for dt in doc_tokens:
            union_doc.update(dt)
        cov = sum(1 for t in q if t in union_doc) / len(q)
    jac = _max_jaccard(q, doc_tokens)

    # Lexical spread: average pairwise Jaccard between passage bags (low = diverse / noisy)
    if len(doc_tokens) >= 2:
        jijs: List[float] = []
        for i in range(len(doc_tokens)):
            si = set(doc_tokens[i])
            if not si:
                continue
            for j in range(i + 1, len(doc_tokens)):
                sj = set(doc_tokens[j])
                if not sj:
                    continue
                u = len(si | sj)
                if u:
                    jijs.append(len(si & sj) / u)
        mean_pair_j = float(np.mean(jijs)) if jijs else 0.0
    else:
        mean_pair_j = 1.0

    prec_max, prec_mean = _query_passage_token_precisions(q, doc_tokens)
    dmax, dmean, dent = _dense_cosine_stats(question, passages, dense_encoder)

    feats = np.array(
        [
            max_b,
            mean_b,
            std_b,
            gap,
            ent,
            math.log1p(n_pass),
            mean_pw,
            math.log1p(total_pw),
            cov,
            jac,
            mean_pair_j,
            math.log1p(float(len(q_raw))),
            min_b,
            med_b,
            prec_max,
            prec_mean,
            dmax,
            dmean,
            dent,
        ],
        dtype=np.float64,
    )
    return feats


class DensePassageEncoder:
    """Same embedding space as ``rag.DocumentRetriever`` (default MiniLM)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def cosine_scores(self, question: str, passages: Sequence[str]) -> np.ndarray:
        passages = [str(p) for p in passages if p is not None and str(p).strip()]
        if not passages or not str(question).strip():
            return np.zeros(0, dtype=np.float64)
        model = self._get_model()
        qe = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        pe = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(pe @ qe[0], dtype=np.float64)


@dataclass
class LexicalAbstentionModel:
    """
    Predict P(ANSWER) = sigmoid(w^T z(x) + b) where z is standardized features.
    Abstain when P(ANSWER) < threshold (default 0.5).
    """

    weights: np.ndarray  # (n_features,)
    bias: float
    mean: np.ndarray
    std: np.ndarray
    embedding_model_name: Optional[str] = None
    use_dense_features: bool = False

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        s = np.where(self.std > 1e-8, self.std, 1.0)
        return (x - self.mean) / s

    def answer_probability(
        self,
        question: str,
        passages: Sequence[str],
        *,
        dense_encoder: Optional[DensePassageEncoder] = None,
    ) -> float:
        enc = (
            dense_encoder
            if (self.use_dense_features and dense_encoder is not None)
            else None
        )
        x = extract_features(question, passages, dense_encoder=enc)
        z = self._standardize(x)
        logit = float(np.dot(self.weights, z) + self.bias)
        logit = max(-50.0, min(50.0, logit))
        return float(1.0 / (1.0 + math.exp(-logit)))

    def should_abstain(
        self,
        question: str,
        passages: Sequence[str],
        threshold: float = 0.5,
        *,
        dense_encoder: Optional[DensePassageEncoder] = None,
    ) -> Tuple[bool, float]:
        p_answer = self.answer_probability(
            question, passages, dense_encoder=dense_encoder
        )
        return p_answer < threshold, p_answer

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_features": int(len(self.weights)),
            "use_dense_features": self.use_dense_features,
        }
        if self.embedding_model_name:
            d["embedding_model"] = self.embedding_model_name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LexicalAbstentionModel:
        w = np.array(d["weights"], dtype=np.float64)
        if w.shape[0] != _N_FEATURES:
            raise ValueError(
                f"Checkpoint has n_features={w.shape[0]} but this code expects {_N_FEATURES}. "
                "Retrain with: python feature_gated_rag.py --fit squad_train.json"
            )
        mean = np.array(d["mean"], dtype=np.float64)
        std = np.array(d["std"], dtype=np.float64)
        if mean.shape[0] != _N_FEATURES or std.shape[0] != _N_FEATURES:
            raise ValueError(
                f"Checkpoint mean/std length does not match weights ({_N_FEATURES} expected)."
            )
        emb = d.get("embedding_model")
        use_d = bool(d.get("use_dense_features", False))
        return cls(
            weights=w,
            bias=float(d["bias"]),
            mean=mean,
            std=std,
            embedding_model_name=str(emb).strip() if emb else None,
            use_dense_features=use_d,
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> LexicalAbstentionModel:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.15,
    epochs: int = 4000,
    l2: float = 5e-2,
    seed: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Fit binary logistic regression; y=1 means ANSWER, y=0 means ABSTAIN.
    Returns weights, bias, feature_mean, feature_std.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    Xn = (X - mean) / std

    w = rng.normal(scale=0.01, size=d)
    b = 0.0
    y = y.astype(np.float64)

    for _ in range(epochs):
        z = Xn @ w + b
        p = _sigmoid(z)
        err = (p - y) / n
        grad_w = Xn.T @ err + l2 * w
        grad_b = float(err.sum())
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mean, std


def _decision_to_label(decision: str) -> Optional[int]:
    d = str(decision).strip().upper()
    if d == "ANSWER":
        return 1
    if d == "ABSTAIN":
        return 0
    return None


def build_training_matrix(
    examples: Sequence[Dict[str, Any]],
    dense_encoder: Optional[DensePassageEncoder] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Stack features and labels from example dicts with ``decision`` + question + retrieved passages."""
    rows: List[np.ndarray] = []
    labels: List[int] = []
    kept_idx: List[int] = []
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            continue
        lab = _decision_to_label(str(ex.get("decision", "")))
        if lab is None:
            continue
        q = str(ex.get("question", ""))
        ctx = ex.get("retrieved_context")
        if ctx is None:
            ctx = ex.get("retrieved_passages")
        if isinstance(ctx, str):
            passages = [ctx] if ctx.strip() else []
        elif isinstance(ctx, list):
            passages = [str(p) for p in ctx if p is not None and str(p).strip()]
        else:
            passages = []
        rows.append(extract_features(q, passages, dense_encoder=dense_encoder))
        labels.append(lab)
        kept_idx.append(i)
    if not rows:
        raise ValueError("No training rows with decision ANSWER/ABSTAIN and a question.")
    return np.stack(rows, axis=0), np.array(labels, dtype=np.float64), kept_idx


def train_from_labeled_examples(
    examples: Sequence[Dict[str, Any]],
    *,
    dense_encoder: Optional[DensePassageEncoder] = None,
    use_dense_features: bool = True,
    **kwargs: Any,
) -> LexicalAbstentionModel:
    enc = dense_encoder if use_dense_features and dense_encoder is not None else None
    X, y, _ = build_training_matrix(examples, dense_encoder=enc)
    kwargs.setdefault("l2", 5e-2)
    w, b, mean, std = fit_logistic_regression(X, y, **kwargs)
    use_d = enc is not None
    emb_name = dense_encoder.model_name if dense_encoder is not None and use_d else None
    return LexicalAbstentionModel(
        weights=w,
        bias=b,
        mean=mean,
        std=std,
        embedding_model_name=emb_name,
        use_dense_features=use_d,
    )
