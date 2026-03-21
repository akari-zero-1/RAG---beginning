import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in directory '{docs_path}'.")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating vector store...")

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    embedding_model = get_embedding_model()

    # Nếu DB đã tồn tại → load lại
    if os.path.exists(persistent_directory):
        print("[OK] Vector store already exists. Loading...")

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n[SUCCESS] Ingestion complete! Documents are ready for RAG.")
    return vectorstore


if __name__ == "__main__":
    main()