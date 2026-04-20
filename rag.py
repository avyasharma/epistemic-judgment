import argparse
import json
import os
import faiss
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from squad_utils import gold_answer_label_for_squad_record, ordered_gold_answers_shortest_first

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

SquadRecord = Dict[str, Any]

print("Loading environment variables...")

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
BASE_URL = os.getenv("BASE_URL")

class OpenAIClient():
    def __init__(self, model: str = "gpt-5-mini"):
        # Initialize OpenAI client (make sure OPENAI_API_KEY is configured)
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        self.model = model

    def generate(self, query: str, context: List[str]) -> str:
        context_str = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])
        
        system_prompt = (
            "You answer reading-comprehension questions using only the provided documents. "
            "Reply with the shortest text that suffices: a name, date, number, or a short "
            "noun phrase (SQuAD-style span), not a full sentence. Do not prefix with "
            "\"The answer is\", do not add explanations or restate the question. "
            "If the documents do not contain the answer, reply exactly: I do not know."
        )

        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer (minimal span only, or I do not know):"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


class DocumentRetriever():
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS Index (L2 distance)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents: List[str] = []
        
    def add_documents(self, documents: List[str], chunk_size: int = 150, chunk_overlap: int = 30):
        if not documents:
            return
            
        chunked_docs = []
        for doc in documents:
            words = doc.split()
            if not words:
                continue
            
            start = 0
            while start < len(words):
                end = int(start + chunk_size)
                chunked_docs.append(" ".join(words[int(start):end]))
                start += int(chunk_size - chunk_overlap)
                
        self.documents.extend(chunked_docs)
        # Convert text to embeddings mapping using the huggingface module
        embeddings = self.embedding_model.encode(chunked_docs, convert_to_numpy=True)
        # Add to FAISS
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        retrieved_docs = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])
        return retrieved_docs
        

class VanillaRAG:
    def __init__(self, retriever: DocumentRetriever, generator: OpenAIClient):
        self.retriever = retriever
        self.generator = generator

    def retrieve_only(self, query: str, top_k: int = 5) -> List[str]:
        return self.retriever.retrieve(query, top_k=top_k)

    def generate_only(self, query: str, passages: List[str]) -> str:
        return self.generator.generate(query, passages)

    def answer(self, query: str, top_k: int = 5) -> tuple[str, List[str]]:
        passages = self.retrieve_only(query, top_k=top_k)
        return self.generate_only(query, passages), passages

# --- Dataset loaders: SQuAD JSON array (from squad_excel_to_json.py or upstream SQuAD) ---


def load_squad_json_records(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {json_path}")
    return data


def squad_records_to_unique_passages(records: List[Dict[str, Any]]) -> List[str]:
    """Flatten context passages across examples; preserve first-seen order."""
    seen = set()
    out: List[str] = []
    for rec in records:
        for p in rec.get("context") or []:
            if not isinstance(p, str):
                p = str(p)
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def squad_record_to_retrieval_entry(rec: Dict[str, Any], passages: List[str]) -> Dict[str, Any]:
    """One row for retrieval-only JSON (before generation)."""
    gold = gold_answer_label_for_squad_record(rec)
    ordered = (
        ordered_gold_answers_shortest_first(rec.get("answers") or [])
        if not rec.get("is_impossible")
        else []
    )
    return {
        "question": rec["question"],
        "gold answer": gold,
        "gold_answers": ordered if ordered else (rec.get("answers") or []),
        "is_impossible": rec.get("is_impossible", False),
        "ids": rec.get("ids"),
        "retrieved_context": passages,
    }


def load_documents_for_retriever(document_path: str) -> Tuple[List[str], Optional[List[SquadRecord]]]:
    """
    Returns (documents_for_indexing, squad_records).

    ``document_path`` must be a JSON file containing a SQuAD-style array of records
    (each with a ``context`` list of passage strings).
    """
    if not document_path.lower().endswith(".json"):
        raise ValueError(
            f"Expected a SQuAD JSON array (.json), got {document_path!r}. "
            "CSV / Kaggle document corpora are no longer supported."
        )
    records = load_squad_json_records(document_path)
    docs = squad_records_to_unique_passages(records)
    print(f"[Dataset] SQuAD JSON: indexed {len(docs)} unique passages from {document_path}")
    return docs, records


def build_rag_pipeline(document_path: str, embedding_model: str, llm_model: str) -> VanillaRAG:
    print("Initializing retriever...")
    retriever = DocumentRetriever(model_name=embedding_model)

    print("Initializing generator...")
    generator = OpenAIClient(model=llm_model)

    print("Loading documents...")
    documents, _ = load_documents_for_retriever(document_path)
    print(f"Loaded {len(documents)} raw documents")

    print("Adding documents to retriever (chunking + embedding + indexing)...")
    retriever.add_documents(documents)
    print(f"Indexed {len(retriever.documents)} chunks")

    print("RAG pipeline ready.")
    return VanillaRAG(retriever, generator)


def _write_json_atomic(path: str, obj: Any) -> None:
    """Write JSON then replace, so readers never see a half-written file."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_retrieval_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected retrieval JSON object, got {type(data).__name__}")
    return data


def _retrieval_covers_records(retrieval: Dict[str, Any], n_records: int) -> bool:
    if len(retrieval) < n_records:
        return False
    for i in range(n_records):
        ex = retrieval.get(str(i))
        if not isinstance(ex, dict):
            return False
        ctx = ex.get("retrieved_context") or ex.get("retrieved_passages")
        if not ctx:
            return False
    return True


def _generation_done(ex: Any) -> bool:
    if not isinstance(ex, dict):
        return False
    r = ex.get("response")
    return r is not None and str(r).strip() != ""


def default_squad_json_path() -> str:
    """``SQUAD_JSON`` env, else ``squad_test_sampled.json`` if it exists, else ``squad_test.json``."""
    if os.getenv("SQUAD_JSON"):
        return os.environ["SQUAD_JSON"]
    sampled = "squad_test_sampled.json"
    if os.path.isfile(sampled):
        return sampled
    return "squad_test.json"


def resolve_squad_json_path_for_cli(explicit: Optional[str]) -> str:
    """CLI / env resolution for ``rag.py`` main: explicit ``--squad-json``, then ``SQUAD_JSON``, else sampled."""
    if explicit:
        return explicit
    if os.getenv("SQUAD_JSON"):
        return os.environ["SQUAD_JSON"]
    return "squad_test_sampled.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Index SQuAD, retrieve, then generate RAG outputs.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If retrieval JSON already has one entry per record, skip retrieval. "
            "Load partial RAG outputs if present and only run LLM for rows missing a non-empty response."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        metavar="N",
        help="Write RAG outputs JSON to disk every N newly completed answers (crash / sleep safety).",
    )
    parser.add_argument(
        "--squad-json",
        default=None,
        metavar="PATH",
        help=(
            "SQuAD-style JSON array (corpus + questions). "
            "When omitted: use SQUAD_JSON if set, else squad_test_sampled.json."
        ),
    )
    args = parser.parse_args()

    json_path = resolve_squad_json_path_for_cli(args.squad_json)
    retrieval_path = os.getenv("RETRIEVAL_JSON", "squad_retrieval.json")
    out_path = os.getenv("RAG_OUTPUTS_JSON", "squad_rag_outputs.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"{json_path!r} not found. Create a 20%% test sample with:\n"
            f"  python squad_utils.py --sample-test\n"
            f"or pass a different corpus: python rag.py --squad-json squad_test.json\n"
            f"or set SQUAD_JSON."
        )

    records = load_squad_json_records(json_path)
    n = len(records)
    rag_pipeline = build_rag_pipeline(
        document_path=json_path,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-5-mini",
    )

    retrieval_outputs: Dict[str, Any] = {}
    skip_retrieval = False
    if args.resume and os.path.isfile(retrieval_path):
        loaded = _load_retrieval_dict(retrieval_path)
        if _retrieval_covers_records(loaded, n):
            retrieval_outputs = loaded
            skip_retrieval = True
            print(f"[resume] Using existing retrieval ({n} rows) from {retrieval_path!r}")
        else:
            print(
                f"[resume] Retrieval file incomplete or mismatched ({len(loaded)} vs {n} records); "
                "re-running retrieval."
            )

    if not skip_retrieval:
        retrieval_outputs = {}
        for i, rec in tqdm(
            enumerate(records),
            total=n,
            desc="Retrieval",
            unit="q",
        ):
            q = rec["question"]
            passages = rag_pipeline.retrieve_only(q, top_k=5)
            retrieval_outputs[str(i)] = squad_record_to_retrieval_entry(rec, passages)

        _write_json_atomic(retrieval_path, retrieval_outputs)
        print(f"Wrote retrieval-only JSON ({len(retrieval_outputs)} examples) to {retrieval_path}")

    outputs: Dict[str, Any] = {}
    if args.resume and os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                outputs = {k: v for k, v in prev.items() if isinstance(v, dict)}
                done = sum(1 for i in range(n) if _generation_done(outputs.get(str(i))))
                print(f"[resume] Loaded {len(outputs)} keys from {out_path!r} ({done}/{n} with responses)")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[resume] Could not load {out_path!r} ({e}); starting generation from scratch.")

    pending = [i for i in range(n) if not _generation_done(outputs.get(str(i)))]
    if not pending:
        print("Generation already complete; nothing to do.")
        _write_json_atomic(out_path, outputs)
        return

    since_flush = 0
    for i in tqdm(pending, desc="Generation", unit="q"):
        rec = records[i]
        q = rec["question"]
        base = retrieval_outputs[str(i)]
        passages = list(base["retrieved_context"])
        response = rag_pipeline.generate_only(q, passages)
        outputs[str(i)] = {**base, "response": response}
        since_flush += 1
        if since_flush >= args.flush_every:
            _write_json_atomic(out_path, outputs)
            since_flush = 0

    _write_json_atomic(out_path, outputs)
    print(f"Wrote {len(outputs)} examples to {out_path}")


if __name__ == "__main__":
    main()
