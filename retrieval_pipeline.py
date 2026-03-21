from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

persistent_directory = 'db/chroma_db'
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata= {"hnsw:space": "cosine"}
)

query = "who is CEO of Tesla? "

retrieval = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs={"k": 3, 
                   "score_threshold": 0.4
                   }
    )
relevant_docs = retrieval.invoke(query)

print(f"Query: {query}")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")