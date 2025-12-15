import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FREE local embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# ----------------- ENV -----------------
load_dotenv()

PDF_PATH = "islr.pdf"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)


# ----------------- HELPERS (TRACED) -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(splits, embeddings)


# ----------------- CACHE KEY -----------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": "all-MiniLM-L6-v2",
        "format": "v1",
    }
    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ----------------- INDEX LOAD / BUILD (TRACED) -----------------
@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap)
    vs = build_vectorstore(splits)

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))

    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": "all-MiniLM-L6-v2"
    }, indent=2))

    return vs


def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap)
    index_dir = INDEX_ROOT / key

    if index_dir.exists() and not force_rebuild:
        return load_index_run(index_dir)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap)


# ----------------- LLM + PROMPT -----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ----------------- FULL PIPELINE (TRACED) -----------------
@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    force_rebuild: bool = False,
):
    vectorstore = load_or_build_index(
        pdf_path,
        chunk_size,
        chunk_overlap,
        force_rebuild
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["rag", "pdf", "gemini"],
            "metadata": {"k": 4}
        }
    )


# ----------------- CLI -----------------
if __name__ == "__main__":
    print("📄 PDF RAG (cached + traced) ready. Ask a question.")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
