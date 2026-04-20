# epistemic-judgment

SQuAD-style JSON arrays (`squad_train.json`, `squad_test.json`, or files from `squad_excel_to_json.py`) supply passages and questions.

## Paths and defaults

- **`default_squad_json_path()`** (`app.py` / helpers): `SQUAD_JSON` if set; else **`squad_test_sampled.json`** if present; else **`squad_test.json`**.
- **`rag.py` CLI**: **`--squad-json PATH`** selects the SQuAD array; if omitted, uses **`SQUAD_JSON`** or defaults to **`squad_test_sampled.json`** (even when that file is missing you get a clear error + how to build it).
- **20% test sample**: `python squad_utils.py --sample-test` writes **`squad_test_sampled.json`** (from `squad_test.json` by default, seed 42). RAG then defaults to this corpus when the file is present.
- **Retrieval step** (`rag.py`): writes **`RETRIEVAL_JSON`** (default **`squad_retrieval.json`**) with questions, gold fields, and **`retrieved_context`** only; then writes **`RAG_OUTPUTS_JSON`** (default **`squad_rag_outputs.json`**) after generation. Progress uses **tqdm**; generation is flushed to disk every **25** answers by default (`--flush-every`).
- **Sleep / crash**: If the OS only **suspended** the machine, the Python process usually **resumes** and continues. If the process **exited**, use **`python rag.py --resume`**: skips retrieval when **`squad_retrieval.json`** is complete, reloads partial **`squad_rag_outputs.json`**, and only generates missing rows (same `SQUAD_JSON` / record order as before).
- **Epistemic gated** (`epistemic_gated_rag.py`): if **`RETRIEVAL_JSON`** / `squad_retrieval.json` exists, loads it and runs judge + generate with **no second retrieval**; otherwise indexes `default_squad_json_path()` and retrieves as before. Output: **`GATED_OUTPUTS_JSON`** (default **`squad_gated_outputs.json`**).
- **Lexical gate** (`feature_gated_rag.py`): default **`--inputs squad_retrieval.json`**. Train: `python feature_gated_rag.py --fit` on `squad_train.json`.

## Typical flow

```bash
python squad_utils.py --sample-test       # optional: squad_test_sampled.json
python rag.py                             # squad_retrieval.json + squad_rag_outputs.json
python rag.py --resume                    # after interruption, if retrieval JSON is done
python epistemic_gated_rag.py             # uses squad_retrieval.json when present
python feature_gated_rag.py               # reads squad_retrieval.json by default
```

## Editing `app.py` (Streamlit UI)

`app.py` is the **Streamlit** demo only. It is safe to change layout, labels, sliders, and how results are displayed. Keep **long-running / batch** work in the other scripts unless you intentionally move logic here.

### Run the app

```bash
streamlit run app.py
```

From the repo root (same directory as `app.py`). Requires the same `.env` as `rag.py` (**`OPENAI_API_KEY`**, optional **`BASE_URL`**) because the pipeline calls the OpenAI-compatible API.

### What `app.py` does today

- Imports **`build_rag_pipeline`** and **`default_squad_json_path`** from **`rag.py`** (not from `epistemic_gated_rag.py`).
- On first **Ask**, it builds a full RAG pipeline against **`default_squad_json_path()`** (see **Paths and defaults** above: env `SQUAD_JSON`, then `squad_test_sampled.json`, then `squad_test.json`).
- **`st.session_state.rag_pipeline`** caches that pipeline for the session so indexing runs **once** per browser session, not on every question.

### Good places for a partner to edit

| Area | File | Notes |
|------|------|--------|
| Page title, layout, copy | `app.py` | `st.set_page_config`, `st.title`, `st.write`, sidebar |
| Retrieval depth | `app.py` | `top_k` slider passed into `answer(..., top_k=...)` |
| Corpus path | Prefer env | Set **`SQUAD_JSON`** so `default_squad_json_path()` picks your file without hard-coding in `app.py` |
| Embedding / LLM model names | `app.py` | Arguments to `build_rag_pipeline(...)` (lines ~29–32) |

### If you want epistemic gating or a pre-built index in the UI

Do **not** duplicate judge logic in `app.py`. Import and call **`serve_epistemic_request`** from **`epistemic_gated_rag.py`** (see that file’s docstring and `--question` / `--index-dir` CLI). Typical pattern:

1. One-time (or CI): **`python build_retriever_index.py ./your_index_dir`** — merges train/test passages by default.
2. In `app.py`: call `serve_epistemic_request(question=..., index_dir="./your_index_dir", ...)` and render the returned JSON fields (`decision`, `justification`, `relevance_score`, …, `response`, `retrieved_context`).

That keeps **`app.py`** as presentation and **`epistemic_gated_rag.py`** / **`rag.py`** as the source of truth for behavior.

### Dependencies for Streamlit

`streamlit` is listed in **`requirements.txt`**. Install with the rest of the project (e.g. `pip install -r requirements.txt`) in the same environment you use for `rag.py`.
