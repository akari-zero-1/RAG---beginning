from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file before running this script.")

persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db  = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)


chat_history = []

def ask_question(user_question):
    print(f"User Question: {user_question}")
    if chat_history:
        
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Rewritten Question for Search: {search_question}")
    else:
        search_question = user_question
    
    retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5  
    }
    )
    
    docs = retriever.invoke(search_question)
    print(f"Retrieved {len(docs)} documents.")
    
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
        
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    messages = [SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history.")] \
           + chat_history \
           + [HumanMessage(content=combined_input)]
    
    result = model.invoke(messages)
    answer = result.content
    
    chat_history.append(HumanMessage(content = user_question))
    chat_history.append(AIMessage(content = answer))
    
    print(f"Answer: {answer}\n")
    return answer

def start_chat():
    print("Welcome to the RAG Chat! Ask your questions based on the ingested documents.")
    print("\nType 'quit' to end the chat.\n")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "quit":
            print("Ending chat. Goodbye!")
            break
        ask_question(user_input)
        
if __name__ == "__main__":
    start_chat()