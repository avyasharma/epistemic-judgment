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
