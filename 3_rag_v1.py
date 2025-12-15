# pip install -U langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf python-dotenv google-generativeai sentence-transformers

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ❌ Gemini embeddings hata diye
# ✅ HF embeddings use karenge (FREE)
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# Optional: LangSmith project
os.environ["LANGCHAIN_PROJECT"] = "RAG-Gemini-Demo"

load_dotenv()  # expects GEMINI_API_KEY in .env

PDF_PATH = "islr.pdf"

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
splits = splitter.split_documents(docs)

# 3) Embeddings + FAISS (HF – FREE, LOCAL)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(splits, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("📄 PDF RAG (HF Embeddings + Gemini) ready. Ask a question (Ctrl+C to exit).")

while True:
    try:
        q = input("\nQ: ").strip()
        if not q:
            continue
        ans = chain.invoke(q)
        print("\nA:", ans)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        break
