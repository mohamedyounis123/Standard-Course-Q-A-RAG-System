# Standard Course Q&A RAG System

An end-to-end Retrieval-Augmented Generation system that answers student questions using only the content of
uploaded course materials (PDF / CSV / DOCX / TXT), across multiple courses, with full source attribution.

The project is split into **two independent pieces**, as requested:

| File | Role | Where it runs |
|---|---|---|
| `course_rag_notebook.ipynb` | Discovers files, extracts/cleans/chunks text, builds embeddings, indexes a persistent vector DB, runs retrieval + answer evaluation and a hallucination test | Kaggle Notebook |
| `app.py` | Loads the already-built vector DB + config and serves a Gradio Q&A UI | Locally, or any machine with Python |

The notebook never imports `app.py` and `app.py` never re-runs any ingestion — they only share two files on disk:
`vector_db/` (the persisted Chroma database) and `rag_outputs/config.json` (the settings the DB was built with).

## 1. Run the notebook on Kaggle

1. Create a new Kaggle Notebook and upload `course_rag_notebook.ipynb`.
2. (Optional) Attach a Kaggle Dataset containing your real course materials under `/kaggle/input/`. Organize them
   in folders named after each course (or include a course-identifying keyword in the filename) so the automatic
   course-detection step works well — e.g. `/kaggle/input/my-courses/Machine_Learning/lecture_01.pdf`.
   * If you don't attach anything, the notebook automatically generates a small demo dataset for 3 courses
     (Machine Learning, Database Systems, Computer Networks) so it still runs end-to-end.
3. Run all cells top to bottom. This builds and evaluates the knowledge base and writes:
   * `/kaggle/working/vector_db/` — the persistent vector database
   * `/kaggle/working/rag_outputs/config.json` — the config the app needs
   * `/kaggle/working/rag_outputs/evaluation_results.json` — retrieval/answer/hallucination metrics
4. Download `vector_db/` and `rag_outputs/config.json` from the Kaggle "Output" tab (or via the Kaggle API).

## 2. Run the app locally

```bash
pip install -r requirements.txt
```

Place the downloaded files next to `app.py` like this:

```
project/
├── app.py
├── requirements.txt
├── rag_outputs/
│   └── config.json
└── vector_db/
    └── ... (chroma files)
```

Then:

```bash
python app.py
```

Gradio will print a local URL (and, if you set `demo.launch(share=True)` in `app.py`, a public one).

### Environment variables (all optional)

| Variable | Purpose | Default |
|---|---|---|
| `RAG_CONFIG_PATH` | Explicit path to `config.json` | auto-detected (`./rag_outputs/config.json`) |
| `RAG_VECTOR_DB_PATH` | Explicit path to the vector DB folder | taken from `config.json`, falls back to `./vector_db` |
| `RAG_LLM_BACKEND` | `local` (default, no key needed) or `api` | `local` |
| `OPENAI_API_KEY` | Only required if `RAG_LLM_BACKEND=api` | — |

Never commit an API key to source control — always set it as an environment variable or a Kaggle Secret.

## 3. What "grounded" means here

Every answer is generated from a strict prompt that instructs the model to use **only** the retrieved course
chunks, to cite which retrieved item(s) it used, and to say
`"I couldn't find this information in the provided course materials."` when the answer isn't present in the
retrieved context — instead of guessing. The notebook includes a dedicated hallucination test (Section 23) that
verifies this behavior on out-of-domain questions.

## 4. Re-running after adding new course materials

Re-run the notebook after uploading new/updated files. The vector DB is only rebuilt if it is currently empty
(Section 13), so if you want to fully re-index from scratch, delete `/kaggle/working/vector_db/` first, then
re-run.
# Standard-Course-Q-A-RAG-System
