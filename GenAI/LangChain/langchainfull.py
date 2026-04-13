import os
import re
from typing import Optional

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ------------------------------
# 1) Environment + Model
# ------------------------------
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
)


# ------------------------------
# 2) Tool Module
# ------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# ------------------------------
# 3) Knowledge Base for RAG
# ------------------------------
def build_retriever():
    docs = [
        Document(page_content="LangChain is a framework for building applications with LLMs."),
        Document(page_content="RAG means Retrieval-Augmented Generation: retrieve context, then generate answer."),
        Document(page_content="Embeddings are vector representations of text used for semantic search."),
        Document(page_content="FAISS is a vector database library used for similarity search."),
        Document(page_content="Memory helps chatbots remember previous conversation turns."),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    splits = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever()


# ------------------------------
# 4) Build Conversational RAG Chain
# ------------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and latest user question, rewrite it as a standalone question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the provided context to answer the question. If context is insufficient, say you do not know.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


# ------------------------------
# 5) Memory by Session
# ------------------------------
_session_store = {}


def get_session_history(session_id: str):
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# ------------------------------
# 6) Pipeline Router: Tool first, then RAG
# ------------------------------
def try_math_tool(user_input: str) -> Optional[str]:
    match = re.search(r"(\d+)\s*\*\s*(\d+)", user_input)
    if not match:
        return None

    a, b = int(match.group(1)), int(match.group(2))
    result = multiply.invoke({"a": a, "b": b})
    return f"[Tool:multiply] {a} * {b} = {result}"


def run_pipeline(user_input: str, session_id: str = "default") -> str:
    tool_answer = try_math_tool(user_input)
    if tool_answer is not None:
        return tool_answer

    result = conversational_rag.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return result["answer"]


# ------------------------------
# 7) CLI Runner
# ------------------------------
def main():
    print("LangChain Full Pipeline")
    print("Model: gemini-2.5-flash-lite")
    print("Type 'exit' to stop.\n")

    session_id = "demo-session"

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Assistant: Please enter a question.")
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        try:
            answer = run_pipeline(user_input, session_id=session_id)
            print(f"Assistant: {answer}\n")
        except Exception as exc:
            print(f"Assistant: Error: {exc}\n")


if __name__ == "__main__":
    main()
