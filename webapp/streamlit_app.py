"""GenRAG - ask questions about any PDF (Streamlit web app).

Retrieval-augmented generation built from scratch (no LangChain, no vector DB):
PDF -> spaCy sentence split -> 10-sentence chunks -> all-mpnet-base-v2 embeddings
-> dot-product top-k retrieval -> Gemini answer grounded in, and citing, the retrieved chunks.
"""

import hashlib
import os
import re
from datetime import datetime, timezone

import pymupdf
import requests
import streamlit as st
import torch
from google import genai
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English

SENTENCES_PER_CHUNK = 10
MIN_CHUNK_TOKENS = 30      # drop tiny chunks (headers, page numbers)
MAX_PAGES = 150            # keep memory/CPU within the free tier
MAX_MB = 20
TOP_K = 5
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "openai/gpt-oss-120b"   # override with a GROQ_MODEL secret if Groq retires it

st.set_page_config(page_title="GenRAG – Ask your PDF", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first visit only)…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-mpnet-base-v2", device="cpu")


@st.cache_resource
def load_nlp():
    nlp = English()
    nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000
    return nlp


def get_secret(name: str, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name) or default


@st.cache_resource
def load_gemini():
    key = get_secret("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


@st.cache_resource
def usage_counter() -> dict:
    return {"queries": 0, "documents": 0}


def _clean(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text.replace("\n", " ").strip())
    return re.sub(r"\.([A-Z])", r". \1", text)


@st.cache_resource(max_entries=10, show_spinner="Reading and indexing the PDF…")
def build_index(file_hash: str, _pdf_bytes: bytes) -> dict:
    nlp, embedder = load_nlp(), load_embedder()
    doc = pymupdf.open(stream=_pdf_bytes, filetype="pdf")
    chunks = []
    for page_no, page in enumerate(doc):
        if page_no >= MAX_PAGES:
            break
        sentences = [s.text.strip() for s in nlp(_clean(page.get_text())).sents if s.text.strip()]
        for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
            text = _clean(" ".join(sentences[i:i + SENTENCES_PER_CHUNK]))
            if len(text) / 4 > MIN_CHUNK_TOKENS:
                chunks.append({"page": page_no + 1, "text": text})
    if not chunks:
        return {"chunks": [], "embeddings": None, "pages": len(doc)}
    embeddings = embedder.encode([c["text"] for c in chunks], batch_size=32,
                                 convert_to_tensor=True, show_progress_bar=False)
    usage_counter()["documents"] += 1
    return {"chunks": chunks, "embeddings": embeddings, "pages": min(len(doc), MAX_PAGES)}


def retrieve(index: dict, query: str, k: int = TOP_K):
    q = load_embedder().encode(query, convert_to_tensor=True)
    scores = util.dot_score(q, index["embeddings"])[0]
    top = torch.topk(scores, k=min(k, len(index["chunks"])))
    return [(float(s), index["chunks"][int(i)]) for s, i in zip(top.values, top.indices)]


def _ask_gemini(prompt: str) -> str:
    client = load_gemini()
    if client is None:
        raise RuntimeError("no GEMINI_API_KEY")
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if not resp.text:
        raise RuntimeError("empty response")
    return resp.text


def _ask_groq(prompt: str) -> str:
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("no GROQ_API_KEY")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": get_secret("GROQ_MODEL", GROQ_MODEL),
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    text = r.json()["choices"][0]["message"].get("content")
    if not text:
        raise RuntimeError("empty response")
    return text


# Try providers in order; fall back to the next one on any failure (quota, rate limit, outage).
PROVIDERS = [("Gemini", _ask_gemini), ("Groq", _ask_groq)]


def generate(query: str, hits) -> tuple[str, str | None]:
    context = "\n\n".join(f"[{n}] (page {c['page']}) {c['text']}" for n, (_, c) in enumerate(hits, 1))
    prompt = f"""You answer questions about a document using ONLY the numbered passages below.
- Cite the passages you used like [1] or [2][3].
- If the passages don't contain the answer, say "I couldn't find that in the document." Do not guess.
- Be concise.

PASSAGES:
{context}

QUESTION: {query}"""
    errors = []
    for name, ask in PROVIDERS:
        try:
            return ask(prompt), name
        except Exception as e:
            msg = str(e).replace("\n", " ")[:160]
            print(f"[llm] {name} failed: {type(e).__name__}: {msg}", flush=True)
            errors.append(f"{name}: {msg}")
    return ("_Answer generation is unavailable right now ("
            + "; ".join(errors) + "). The retrieved passages are shown below._"), None


st.title("📄 GenRAG – Ask questions about any PDF")
st.caption(
    "Retrieval-augmented generation built from scratch, without LangChain or a vector database: "
    "spaCy sentence chunking, `all-mpnet-base-v2` embeddings, dot-product retrieval and an LLM (Gemini, with automatic fallback to Groq), "
    "with answers cited back to the passages they came from. "
    "[Source code](https://github.com/Istm-27/Mygenragproject)"
)

left, right = st.columns([1, 2], gap="large")
with left:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    with st.form("ask", clear_on_submit=False):
        query = st.text_area("Your question", placeholder="e.g. What is the main contribution of this paper?", height=90)
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

with right:
    if submitted:
        if uploaded is None:
            st.warning("Upload a PDF first.")
        elif not query.strip():
            st.warning("Type a question.")
        elif uploaded.size > MAX_MB * 1024 * 1024:
            st.warning(f"Please upload a PDF under {MAX_MB} MB.")
        else:
            data = uploaded.getvalue()
            index = build_index(hashlib.sha256(data).hexdigest(), data)
            if not index["chunks"]:
                st.error("Couldn't extract text from this PDF (it may be a scanned image).")
            else:
                hits = retrieve(index, query.strip())
                with st.spinner("Generating the answer…"):
                    ans, provider = generate(query.strip(), hits)
                usage = usage_counter()
                usage["queries"] += 1
                print(f"[usage] {datetime.now(timezone.utc).isoformat()} queries={usage['queries']} docs={usage['documents']}", flush=True)

                st.subheader("Answer")
                st.markdown(ans)
                st.caption(f"Indexed {index['pages']} pages into {len(index['chunks'])} chunks."
                           + (f" Answer by {provider}." if provider else ""))
                st.subheader("Retrieved passages")
                for n, (score, c) in enumerate(hits, 1):
                    with st.expander(f"[{n}] Page {c['page']} · similarity {score:.3f}", expanded=(n == 1)):
                        st.write(c["text"])
    else:
        st.info("Upload a PDF on the left, type a question and press **Ask**.")
