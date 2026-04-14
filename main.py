import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "mediumblog_rag")

print("Initializing components...")

embeddings = OpenAIEmbeddings()

retriever = None


def select_vectorstore():
    """Match ingestion: Pinecone cloud or local persisted Chroma."""
    menu = (
        "Select vector store (must match how you ran ingestion.py):\n"
        "  1 - Pinecone\n"
        f"  2 - Chroma (local: {CHROMA_PERSIST_DIRECTORY})\n"
        "Enter 1 or 2: "
    )
    while True:
        choice = input(menu).strip().lower()
        if choice in ("1", "pinecone", "p"):
            return PineconeVectorStore(
                index_name=os.environ["INDEX_NAME"], embedding=embeddings
            )
        if choice in ("2", "chroma", "c", "local"):
            return Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
        print("Invalid choice. Enter 1 or 2.")

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def select_chat_llm():
    """Prompt for OpenAI or one of several Ollama chat models."""
    menu = (
        "Select chat model:\n"
        "  1 - OpenAI (ChatOpenAI)\n"
        "  2 - Ollama: qwen3.5:27b\n"
        "  3 - Ollama: gemma3:12b\n"
        "  4 - Ollama: llama3.1:8b\n"
        "  5 - Ollama: qwen3.5:0.8b\n"
        "Enter 1-5: "
    )
    while True:
        choice = input(menu).strip().lower()
        if choice in ("1", "openai"):
            return ChatOpenAI()
        if choice in ("2", "qwen", "ollama"):
            return ChatOllama(model="qwen3.5:27b")
        if choice in ("3", "gemma"):
            return ChatOllama(model="gemma3:12b")
        if choice in ("4", "llama"):
            return ChatOllama(model="llama3.1:8b")
        if choice in ("5", "qwen0.8", "0.8b"):
            return ChatOllama(model="qwen3.5:0.8b")
        print("Invalid choice. Enter 1, 2, 3, 4, or 5.")


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    vectorstore = select_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = select_chat_llm()

    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
