import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "mediumblog_rag")
CHROMA_COLLECTION_NAME_LOCAL = os.environ.get(
    "CHROMA_COLLECTION_NAME_LOCAL", "mediumblog_rag_local"
)
PGVECTOR_CONNECTION_STRING = os.environ.get(
    "PGVECTOR_CONNECTION_STRING", "postgresql+psycopg://localhost:5432/vectordb"
)
PGVECTOR_COLLECTION_NAME = os.environ.get("PGVECTOR_COLLECTION_NAME", "mediumblog_rag")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")


def select_ingestion_backend() -> str:
    """Return 'pinecone', 'chroma', or 'pgvector'."""
    menu = (
        "Select where to store embeddings:\n"
        "  1 - Pinecone     (cloud index)        [embedding: OpenAI text-embedding-ada-002]\n"
        f"  2 - Chroma       (local disk)         [embedding: Ollama/{OLLAMA_EMBEDDING_MODEL}]\n"
        f"  3 - PGVector     (local postgres)     [embedding: Ollama/{OLLAMA_EMBEDDING_MODEL}]\n"
        "Enter 1, 2, or 3: "
    )
    while True:
        choice = input(menu).strip().lower()
        if choice in ("1", "pinecone", "p"):
            return "pinecone"
        if choice in ("2", "chroma", "c", "local"):
            return "chroma"
        if choice in ("3", "pgvector", "pg"):
            return "pgvector"
        print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    print("Ingesting...")
    with open("./mediumblog1.txt", encoding="utf-8") as f:
        document = [Document(page_content=f.read(), metadata={"source": "./mediumblog1.txt"})]

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    backend = select_ingestion_backend()

    # Pinecone uses OpenAI embeddings (cloud); Chroma and PGVector use local Ollama embeddings.
    if backend == "pinecone":
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        print(f"Loading Ollama embedding model '{OLLAMA_EMBEDDING_MODEL}'...")
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    print(f"Ingesting into {backend}...")

    if backend == "pinecone":
        PineconeVectorStore.from_documents(
            texts, embeddings, index_name=os.environ["INDEX_NAME"]
        )
    elif backend == "pgvector":
        PGVector.from_documents(
            texts,
            embedding=embeddings,
            connection=PGVECTOR_CONNECTION_STRING,
            collection_name=PGVECTOR_COLLECTION_NAME,
            use_jsonb=True,
        )
    else:
        Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            collection_name=CHROMA_COLLECTION_NAME_LOCAL,
        )

    print("Done.")
