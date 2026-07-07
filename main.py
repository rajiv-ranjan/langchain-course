import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME_LOCAL = os.environ.get(
    "CHROMA_COLLECTION_NAME_LOCAL", "mediumblog_rag_local"
)
PGVECTOR_CONNECTION_STRING = os.environ.get(
    "PGVECTOR_CONNECTION_STRING", "postgresql+psycopg://localhost:5432/vectordb"
)
PGVECTOR_COLLECTION_NAME = os.environ.get("PGVECTOR_COLLECTION_NAME", "mediumblog_rag")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")

print("Initializing components...")

# Module-level globals — set in __main__ and shared across all functions.
vectorstore = None
retriever = None
llm = None


# ============================================================================
# Setup helpers
# ============================================================================

def select_vectorstore():
    """Match ingestion: Pinecone (OpenAI embeddings) or local Chroma/PGVector (Ollama embeddings)."""
    menu = (
        "Select vector store (must match how you ran ingestion.py):\n"
        "  1 - Pinecone     (cloud index)        [embedding: OpenAI text-embedding-ada-002]\n"
        f"  2 - Chroma       (local disk)         [embedding: Ollama/{OLLAMA_EMBEDDING_MODEL}]\n"
        f"  3 - PGVector     (local postgres)     [embedding: Ollama/{OLLAMA_EMBEDDING_MODEL}]\n"
        "Enter 1, 2, or 3: "
    )
    while True:
        choice = input(menu).strip().lower()
        if choice in ("1", "pinecone", "p"):
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            return PineconeVectorStore(
                index_name=os.environ["INDEX_NAME"], embedding=embeddings
            )
        if choice in ("2", "chroma", "c", "local"):
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
            return Chroma(
                collection_name=CHROMA_COLLECTION_NAME_LOCAL,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
        if choice in ("3", "pgvector", "pg"):
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
            return PGVector(
                embeddings=embeddings,
                collection_name=PGVECTOR_COLLECTION_NAME,
                connection=PGVECTOR_CONNECTION_STRING,
                use_jsonb=True,
            )
        print("Invalid choice. Enter 1, 2, or 3.")


def select_score_threshold() -> float | None:
    """Ask user for an optional minimum relevance threshold (0.0–1.0).

    Returns the threshold float, or None if the user skips.
    """
    prompt = (
        "\nSet a minimum relevance threshold to filter retrieved documents.\n"
        "  • Scores are normalized 0.0–1.0 (higher = more relevant).\n"
        "  • Documents below the threshold are discarded before reaching the LLM.\n"
        "  • Suggested starting points: Pinecone → 0.75 | Chroma/PGVector → 0.60\n"
        "Enter threshold (0.0–1.0) or press Enter to skip [no filter]: "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("No threshold set — all top-k documents will be used.")
            return None
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                print(f"Threshold set to {value:.2f} — docs below this score will be discarded.")
                return value
            print("Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Enter a decimal number like 0.7, or press Enter to skip.")


def select_chat_llm():
    """Prompt for OpenAI or one of several Ollama chat models."""
    menu = (
        "Select chat model:\n"
        "  1 - OpenAI (ChatOpenAI)\n"
        "  2 - Ollama: qwen3.5:27b\n"
        "  3 - Ollama: gemma3:12b\n"
        "  4 - Ollama: llama3.2:3b\n"
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
            return ChatOllama(model="llama3.2:3b")
        if choice in ("5", "qwen0.8", "0.8b"):
            return ChatOllama(model="qwen3.5:0.8b")
        print("Invalid choice. Enter 1, 2, 3, 4, or 5.")


# ============================================================================
# Document preview helpers
# ============================================================================

def ask_preview_docs() -> bool:
    """Ask whether to preview retrieved documents and their relevance scores."""
    while True:
        choice = input("\nPreview retrieved documents with relevance scores? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("Enter y or n.")


def print_docs_with_scores(query: str) -> None:
    """Retrieve documents with relevance scores and print a formatted preview.

    Uses similarity_search_with_relevance_scores() which normalises scores to
    0.0–1.0 (higher = more relevant) consistently across all three backends.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    print(f"\n{'─' * 70}")
    print(f"  RETRIEVED DOCUMENTS — {len(results)} result(s) for query:")
    print(f"  \"{query}\"")
    print(f"{'─' * 70}")

    if not results:
        print("  No documents returned.")
        print(f"{'─' * 70}\n")
        return

    for i, (doc, score) in enumerate(results, start=1):
        pct = score * 100
        if pct >= 75:
            label = "High"
        elif pct >= 50:
            label = "Medium"
        else:
            label = "Low"

        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:200].replace("\n", " ").strip()
        if len(doc.page_content) > 200:
            preview += "..."

        print(f"\n  Doc {i} of {len(results)}")
        print(f"  Relevance : {pct:.1f}%  [{label}]")
        print(f"  Source    : {source}")
        print(f"  Preview   : {preview}")

    print(f"\n{'─' * 70}\n")


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
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
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


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# Option 0: Raw invocation without RAG
# ============================================================================
def run_option_0(query: str):
    """Option 0: Raw LLM invocation without RAG."""
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)


# ============================================================================
# Option 1: Use implementation WITHOUT LCEL
# ============================================================================
def run_option_1(query: str):
    """Option 1: Retrieval chain without LCEL."""
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    if ask_preview_docs():
        print_docs_with_scores(query)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)


# ============================================================================
# Option 2: Use implementation WITH LCEL (Better Approach)
# ============================================================================
def run_option_2(query: str):
    """Option 2: Retrieval chain with LCEL (Better Approach)."""
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
    if ask_preview_docs():
        print_docs_with_scores(query)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)


def select_option() -> int:
    """Prompt the user to choose which implementation to run."""
    menu = (
        "\nSelect implementation to run:\n"
        "  0 - Raw LLM invocation (No RAG)\n"
        "  1 - Retrieval chain without LCEL\n"
        "  2 - Retrieval chain with LCEL (recommended)\n"
        "Enter 0, 1, or 2: "
    )
    while True:
        choice = input(menu).strip()
        if choice in ("0", "1", "2"):
            return int(choice)
        print("Invalid choice. Enter 0, 1, or 2.")


prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

if __name__ == "__main__":
    vectorstore = select_vectorstore()

    threshold = select_score_threshold()
    if threshold is not None:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": threshold, "k": 3},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = select_chat_llm()

    print("\nType 'exit' or 'quit' at any prompt to stop.\n")

    while True:
        DEFAULT_QUERY = "what is Pinecone in machine learning?"
        query = input(f"Enter your query [{DEFAULT_QUERY}]: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not query:
            query = DEFAULT_QUERY
            print(f"Using default query: {query}")

        option = select_option()

        if option == 0:
            run_option_0(query)
        elif option == 1:
            run_option_1(query)
        elif option == 2:
            run_option_2(query)
