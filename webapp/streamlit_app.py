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


@st.cache_resource
def load_gemini():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("GEMINI_API_KEY")
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


def generate(query: str, hits) -> str:
    client = load_gemini()
    if client is None:
        return "_Answer generation is unavailable (no Gemini API key configured). The retrieved passages are shown below._"
    context = "\n\n".join(f"[{n}] (page {c['page']}) {c['text']}" for n, (_, c) in enumerate(hits, 1))
    prompt = f"""You answer questions about a document using ONLY the numbered passages below.
- Cite the passages you used like [1] or [2][3].
- If the passages don't contain the answer, say "I couldn't find that in the document." Do not guess.
- Be concise.

PASSAGES:
{context}

QUESTION: {query}"""
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text or "_The model returned an empty response._"
    except Exception as e:  # rate limit, quota, network
        return f"_Answer generation failed ({type(e).__name__}); the API may be rate-limited. The retrieved passages are shown below._"


st.title("📄 GenRAG – Ask questions about any PDF")
st.caption(
    "Retrieval-augmented generation built from scratch, without LangChain or a vector database: "
    "spaCy sentence chunking, `all-mpnet-base-v2` embeddings, dot-product retrieval and Gemini, "
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
                with st.spinner("Asking Gemini…"):
                    ans = generate(query.strip(), hits)
                usage = usage_counter()
                usage["queries"] += 1
                print(f"[usage] {datetime.now(timezone.utc).isoformat()} queries={usage['queries']} docs={usage['documents']}", flush=True)

                st.subheader("Answer")
                st.markdown(ans)
                st.caption(f"Indexed {index['pages']} pages into {len(index['chunks'])} chunks.")
                st.subheader("Retrieved passages")
                for n, (score, c) in enumerate(hits, 1):
                    with st.expander(f"[{n}] Page {c['page']} · similarity {score:.3f}", expanded=(n == 1)):
                        st.write(c["text"])
    else:
        st.info("Upload a PDF on the left, type a question and press **Ask**.")
