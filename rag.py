import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage


class LitigationRAG:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
        self.vectorstore: Optional[FAISS] = None
        self.doc_name: str = ""

    def load_pdf(self, pdf_path: str) -> int:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.doc_name = os.path.basename(pdf_path)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> dict:
        if not self.vectorstore:
            raise ValueError("No document loaded. Upload and index a PDF first.")

        docs = self.vectorstore.similarity_search(question, k=k)

        context_parts = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", 0) + 1
            context_parts.append(f"[Exhibit {i + 1} — Page {page}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            SystemMessage(content=(
                "You are a legal research assistant specialized in litigation analysis. "
                "Answer questions based ONLY on the provided document excerpts. "
                "Always cite sources using [Exhibit N — Page P] notation inline. "
                "Be precise and professional. If the information is not in the excerpts, say so explicitly."
            )),
            HumanMessage(content=f"Document excerpts:\n\n{context}\n\nQuestion: {question}\n\nAnswer with inline citations:"),
        ]

        response = self.llm.invoke(messages)

        sources = [
            {
                "exhibit": i + 1,
                "page": doc.metadata.get("page", 0) + 1,
                "excerpt": doc.page_content[:350].strip(),
                "source": self.doc_name,
            }
            for i, doc in enumerate(docs)
        ]

        return {"answer": response.content, "sources": sources}
