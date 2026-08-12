"""
Standard Course Q&A RAG System - Gradio App
=============================================

This app is intentionally SEPARATE from the notebook that builds the knowledge base
(`course_rag_notebook.ipynb`). It does not scan input files, extract text, clean, chunk,
or generate embeddings for the corpus - it only:

  1. Loads `config.json` (written by the notebook in Section 27).
  2. Opens the persisted Chroma vector database the notebook already built and indexed.
  3. Loads the same embedding model used to build that database.
  4. Serves a Gradio chat-style UI over the same `answer_question()` RAG pipeline.

Run locally:
    pip install -r requirements.txt
    python app.py

Configuration (all optional, override via environment variables):
    RAG_CONFIG_PATH      Path to config.json produced by the notebook
                          (default: looks in ./rag_outputs/config.json, then
                          /kaggle/working/rag_outputs/config.json)
    RAG_VECTOR_DB_PATH   Overrides the vector_db path stored in config.json - useful
                          when you downloaded the vector_db folder from Kaggle to a
                          different local path.
    RAG_LLM_BACKEND      "local" (default) or "api" - overrides config.json's LLM_BACKEND.
    OPENAI_API_KEY       Required only if RAG_LLM_BACKEND="api". Never hard-code this.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import torch
import gradio as gr

NO_ANSWER_MESSAGE = "I couldn't find this information in the provided course materials."


# --------------------------------------------------------------------------------------
# 1. Load configuration produced by the notebook
# --------------------------------------------------------------------------------------
def _find_config_path() -> Path:
    candidates = [
        os.environ.get("RAG_CONFIG_PATH"),
        "./rag_outputs/config.json",
        "/kaggle/working/rag_outputs/config.json",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not find config.json. Run the notebook first (it writes "
        "/kaggle/working/rag_outputs/config.json), then either:\n"
        "  - copy 'rag_outputs/config.json' and 'vector_db/' next to this app.py, or\n"
        "  - set the RAG_CONFIG_PATH / RAG_VECTOR_DB_PATH environment variables."
    )


CONFIG_PATH = _find_config_path()
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG: Dict[str, Any] = json.load(f)

VECTOR_DB_PATH = os.environ.get("RAG_VECTOR_DB_PATH", CONFIG["VECTOR_DB_PATH"])
if not Path(VECTOR_DB_PATH).exists():
    # Fall back to a vector_db folder placed next to config.json (common after downloading from Kaggle)
    local_candidate = CONFIG_PATH.parent.parent / "vector_db"
    if local_candidate.exists():
        VECTOR_DB_PATH = str(local_candidate)
    else:
        raise FileNotFoundError(
            f"Vector database not found at '{VECTOR_DB_PATH}'. Copy the 'vector_db/' folder produced by "
            f"the notebook next to this app, or set RAG_VECTOR_DB_PATH."
        )

CONFIG["LLM_BACKEND"] = os.environ.get("RAG_LLM_BACKEND", CONFIG.get("LLM_BACKEND", "local"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COURSES: List[str] = CONFIG.get("DETECTED_COURSES") or CONFIG.get("COURSES", [])
COURSE_CHOICES = ["All Courses"] + COURSES

print(f"[app] Loaded config from: {CONFIG_PATH}")
print(f"[app] Vector DB path    : {VECTOR_DB_PATH}")
print(f"[app] Embedding model   : {CONFIG['EMBEDDING_MODEL']}")
print(f"[app] LLM backend       : {CONFIG['LLM_BACKEND']}")
print(f"[app] Courses           : {COURSES}")
print(f"[app] Device            : {DEVICE}")


# --------------------------------------------------------------------------------------
# 2. Load the embedding model and the persisted vector database
# --------------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer(CONFIG["EMBEDDING_MODEL"], device=DEVICE)

chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=CONFIG["COLLECTION_NAME"],
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    print("[app][WARN] The vector database is empty. Run the notebook to build and index the knowledge base "
          "before using this app.")
else:
    print(f"[app] Vector DB ready: {collection.count()} indexed chunks.")


# --------------------------------------------------------------------------------------
# 3. RAG pipeline - retrieval, prompting, generation, source attribution
#    (self-contained here so app.py has no dependency on the notebook's runtime)
# --------------------------------------------------------------------------------------
def retrieve_documents(query: str, top_k: int = 5, course: Optional[str] = None,
                        similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
    if not query or not query.strip() or collection.count() == 0:
        return []
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
    where_filter = {"course": course} if course and course != "All Courses" else None
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_filter)

    retrieved = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            similarity = 1 - results["distances"][0][i]
            if similarity >= similarity_threshold:
                retrieved.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(float(similarity), 4),
                })
    return retrieved


def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        context_block = "(no relevant course material was retrieved)"
    else:
        lines = []
        for i, r in enumerate(retrieved):
            m = r["metadata"]
            page_info = f", page {m['page']}" if m.get("page") not in (None, "", "None") else ""
            lines.append(f"[{i + 1}] ({m['course']} - {m['source']}{page_info})\n{r['text']}")
        context_block = "\n\n".join(lines)

    return f"""Answer the QUESTION using only the information in the CONTEXT below. \
Do not use any outside knowledge, and do not invent facts that are not stated in the CONTEXT.
If the CONTEXT does not contain the answer, respond with exactly this sentence and nothing else:
"{NO_ANSWER_MESSAGE}"

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""


_SUSPICIOUS_ANSWER_RE = re.compile(r"^\s*(\[\d+\]\.?\s*)+$")


def _looks_incomplete(answer: str) -> bool:
    """Detect degenerate local-model outputs (e.g. '[1].') that aren't an actual answer."""
    stripped = answer.strip()
    if len(stripped) < 12:
        return True
    if _SUSPICIOUS_ANSWER_RE.match(stripped):
        return True
    return False


def format_sources(retrieved: List[Dict[str, Any]]) -> List[str]:
    sources = []
    for i, r in enumerate(retrieved):
        m = r["metadata"]
        page_info = f" — Page {m['page']}" if m.get("page") not in (None, "", "None") else ""
        sources.append(f"[{i + 1}] {m['course']} — {m['source']}{page_info} (similarity: {r['similarity']:.2f})")
    return sources


_local_tokenizer = None
_local_model = None


def _get_local_model():
    """Lazily load flan-t5 via AutoTokenizer/AutoModelForSeq2SeqLM.

    We deliberately avoid pipeline("text2text-generation", ...): that task alias existed in
    Transformers v4 but was REMOVED in v5 (raises KeyError: "Unknown task text2text-generation").
    Loading the tokenizer/model classes directly is stable across both major versions.
    """
    global _local_tokenizer, _local_model
    if _local_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _local_tokenizer = AutoTokenizer.from_pretrained(CONFIG["LOCAL_LLM_MODEL"])
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["LOCAL_LLM_MODEL"]).to(DEVICE)
        _local_model.eval()
    return _local_tokenizer, _local_model


def _generate_local(prompt: str, max_new_tokens: int = 300) -> str:
    tokenizer, model = _get_local_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _get_api_key() -> Optional[str]:
    return os.environ.get("OPENAI_API_KEY")


def _generate_api(prompt: str, max_new_tokens: int = 300) -> str:
    api_key = _get_api_key()
    if not api_key:
        print("[app][WARN] LLM_BACKEND='api' but OPENAI_API_KEY is not set - falling back to the local model.")
        return _generate_local(prompt, max_new_tokens)
    import requests
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": CONFIG.get("API_MODEL_NAME", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_answer(prompt: str, max_new_tokens: int = 300) -> str:
    if CONFIG["LLM_BACKEND"] == "api":
        return _generate_api(prompt, max_new_tokens)
    return _generate_local(prompt, max_new_tokens)


def answer_question(question: str, course: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    retrieved = retrieve_documents(question, top_k=top_k, course=course)
    if not retrieved:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "retrieved_documents": []}
    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)
    if _looks_incomplete(answer):
        answer = (
            "The local model produced an incomplete response for this question. This can happen with small "
            "local models on harder questions - try rephrasing, or set RAG_LLM_BACKEND=api for stronger "
            "answers. The retrieved sources below are still accurate and worth checking directly."
        )
    return {"answer": answer, "sources": format_sources(retrieved), "retrieved_documents": retrieved}


# --------------------------------------------------------------------------------------
# 4. Gradio interface
# --------------------------------------------------------------------------------------
def ask(question: str, course: str, top_k: int):
    if not question or not question.strip():
        return "Please enter a question.", ""
    if collection.count() == 0:
        return ("The knowledge base is empty. Run the notebook first to build and index the course "
                "materials, then restart this app."), ""
    result = answer_question(question.strip(), course=course, top_k=int(top_k))
    sources_md = "\n".join(f"- {s}" for s in result["sources"]) if result["sources"] else "_No sources retrieved._"
    return result["answer"], sources_md


EXAMPLE_QUESTIONS = [
    ["What is overfitting?", "All Courses", 5],
    ["What is database normalization?", "Database Systems", 5],
    ["Explain TCP congestion control.", "Computer Networks", 5],
    ["What is supervised learning?", "Machine Learning", 5],
    ["Who won the FIFA World Cup in 2022?", "All Courses", 5],
]

with gr.Blocks(title="Standard Course Q&A RAG System") as demo:
    gr.Markdown(
        "# 📚 Standard Course Q&A RAG System\n"
        "Ask a question about your course materials. Answers are generated **only** from the retrieved "
        "course content, with sources shown below every answer.\n\n"
        f"*Knowledge base: {collection.count()} indexed chunks across {len(COURSES)} course(s) "
        f"({'demo dataset' if CONFIG.get('USING_DEMO_DATA') else 'real course materials'}).*"
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is overfitting?",
                lines=2,
            )
        with gr.Column(scale=1):
            course_dropdown = gr.Dropdown(
                label="Course",
                choices=COURSE_CHOICES,
                value="All Courses",
            )
            top_k_slider = gr.Slider(
                label="Top-K retrieved chunks",
                minimum=1, maximum=10, step=1, value=CONFIG.get("TOP_K", 5),
            )

    ask_button = gr.Button("Ask", variant="primary")

    answer_output = gr.Markdown(label="Answer")
    sources_output = gr.Markdown(label="Sources")

    ask_button.click(
        fn=ask,
        inputs=[question_box, course_dropdown, top_k_slider],
        outputs=[answer_output, sources_output],
    )
    question_box.submit(
        fn=ask,
        inputs=[question_box, course_dropdown, top_k_slider],
        outputs=[answer_output, sources_output],
    )

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=[question_box, course_dropdown, top_k_slider],
        label="Example questions (including one with no answer in the course materials)",
    )


if __name__ == "__main__":
    demo.launch()
