import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "mediumblog_rag")


def select_ingestion_backend() -> str:
    """Return 'pinecone' or 'chroma'."""
    menu = (
        "Select where to store embeddings:\n"
        "  1 - Pinecone (cloud index)\n"
        f"  2 - Chroma (local disk: {CHROMA_PERSIST_DIRECTORY})\n"
        "Enter 1 or 2: "
    )
    while True:
        choice = input(menu).strip().lower()
        if choice in ("1", "pinecone", "p"):
            return "pinecone"
        if choice in ("2", "chroma", "c", "local"):
            return "chroma"
        print("Invalid choice. Enter 1 or 2.")


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    backend = select_ingestion_backend()
    print("ingesting...")

    if backend == "pinecone":
        PineconeVectorStore.from_documents(
            texts, embeddings, index_name=os.environ["INDEX_NAME"]
        )
    else:
        Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            collection_name=CHROMA_COLLECTION_NAME,
        )

    print("finish")
