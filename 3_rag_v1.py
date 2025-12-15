# pip install -U langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf python-dotenv google-generativeai

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# LangSmith project (optional)
os.environ["LANGCHAIN_PROJECT"] = "RAG-Gemini-Demo"

load_dotenv()  # expects GEMINI_API_KEY in .env

PDF_PATH = "islr.pdf"   # <-- apna PDF yahin rakho

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
splits = splitter.split_documents(docs)

# 3) Embed + index (Gemini embeddings)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
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

# 5) LLM
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
print("📄 PDF RAG (Gemini) ready. Ask a question (Ctrl+C to exit).")

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
