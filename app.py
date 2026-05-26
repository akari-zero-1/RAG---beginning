import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from collections import defaultdict
from pydantic import BaseModel
from typing import List
import os

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & INITIALIZATION
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🤖 RAG Chat Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RAG Chat - Comprehensive Demo")
st.markdown("*Using various RAG techniques from the project*")

# Load environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Add it to .env file")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# INITIALIZE SESSION STATE
# ═══════════════════════════════════════════════════════════════

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ RAG Settings")
    
    # Retrieval Method
    retrieval_method = st.selectbox(
        "🔍 Retrieval Method",
        ["Similarity Search", "Similarity + Score Threshold", "MMR (Max Marginal Relevance)", "Multi-Query + RRF"],
        help="Choose how documents are retrieved"
    )
    
    # Number of documents
    k = st.slider("📊 Number of Documents (k)", min_value=1, max_value=10, value=3)
    
    # Score threshold (for threshold method)
    if retrieval_method == "Similarity + Score Threshold":
        score_threshold = st.slider("📈 Score Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    else:
        score_threshold = 0.5
    
    # MMR lambda (diversity vs relevance)
    if retrieval_method == "MMR (Max Marginal Relevance)":
        lambda_mult = st.slider("⚖️ Lambda (0=diverse, 1=relevant)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    else:
        lambda_mult = 0.5
    
    st.divider()
    
    # LLM Settings
    st.subheader("🧠 LLM Settings")
    llm_model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        help="Groq LLM model"
    )
    
    temperature = st.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    st.divider()
    
    # Features
    st.subheader("✨ Features")
    use_history_rewrite = st.checkbox("📝 Rewrite query from history", value=True, help="Use chat history to rewrite queries")
    show_retrieved_docs = st.checkbox("📄 Show retrieved documents", value=True)
    use_multi_query = st.checkbox("🔄 Multi-Query Retrieval", value=False, help="Generate query variations")
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", key="clear_history"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.retrieved_docs = []
        st.success("✅ Chat history cleared!")

# ═══════════════════════════════════════════════════════════════
# SETUP LLM & VECTOR DB
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_rag_components():
    persistent_directory = "db/chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    return db, embeddings

db, embeddings = load_rag_components()

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def rewrite_query_from_history(user_question, chat_history):
    """Rewrite question to be standalone using chat history"""
    if not chat_history:
        return user_question
    
    model = ChatGroq(model=llm_model, api_key=groq_api_key, temperature=0)
    
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question, nothing else."),
    ] + chat_history[-4:] + [
        HumanMessage(content=f"New question: {user_question}")
    ]
    
    result = model.invoke(messages)
    return result.content.strip()

def retrieve_similarity(query, k, score_threshold):
    """Basic similarity search with optional score threshold"""
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    return retriever.invoke(query)

def retrieve_mmr(query, k, lambda_mult):
    """Maximum Marginal Relevance retrieval"""
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 2, 10), "lambda_mult": lambda_mult}
    )
    return retriever.invoke(query)

def generate_query_variations(user_question):
    """Generate multiple query variations using LLM"""
    model = ChatGroq(model=llm_model, api_key=groq_api_key, temperature=0.7)
    
    class QueryVariations(BaseModel):
        queries: List[str]
    
    llm_with_tools = model.with_structured_output(QueryVariations)
    
    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {user_question}

Return 3 alternative queries that rephrase or approach the same question from different angles."""
    
    response = llm_with_tools.invoke(prompt)
    return response.queries

def reciprocal_rank_fusion(chunk_lists, k=60):
    """Apply RRF to combine multiple retrieval results"""
    rrf_scores = defaultdict(float)
    all_unique_chunks = {}
    
    for chunk_list in chunk_lists:
        for rank, chunk in enumerate(chunk_list, 1):
            chunk_key = chunk.page_content
            rrf_scores[chunk_key] += 1.0 / (k + rank)
            all_unique_chunks[chunk_key] = chunk
    
    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [all_unique_chunks[chunk_key] for chunk_key, _ in sorted_results]

def generate_answer(user_question, retrieved_docs):
    """Generate answer using LLM"""
    model = ChatGroq(model=llm_model, api_key=groq_api_key, temperature=temperature)
    
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in retrieved_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history.")
    ] + st.session_state.chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    return result.content

# ═══════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════

# Chat input
user_input = st.chat_input("💬 Ask a question about the documents...")

if user_input:
    # Rewrite query if history available
    search_query = user_input
    rewritten_flag = False
    
    if use_history_rewrite and st.session_state.chat_history:
        search_query = rewrite_query_from_history(user_input, st.session_state.chat_history)
        if search_query != user_input:
            rewritten_flag = True
    
    # Retrieve documents
    with st.spinner("🔍 Retrieving documents..."):
        if retrieval_method == "Similarity Search":
            retrieved_docs = retrieve_similarity(search_query, k, 0.0)
        
        elif retrieval_method == "Similarity + Score Threshold":
            retrieved_docs = retrieve_similarity(search_query, k, score_threshold)
        
        elif retrieval_method == "MMR (Max Marginal Relevance)":
            retrieved_docs = retrieve_mmr(search_query, k, lambda_mult)
        
        elif retrieval_method == "Multi-Query + RRF":
            query_variations = generate_query_variations(user_input)
            all_results = []
            for variation in query_variations:
                docs = retrieve_similarity(variation, k, score_threshold)
                all_results.append(docs)
            retrieved_docs = reciprocal_rank_fusion(all_results, k=60)[:k]
    
    # Generate answer
    with st.spinner("🧠 Generating answer..."):
        answer = generate_answer(user_input, retrieved_docs)
    
    # Update history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.retrieved_docs = retrieved_docs
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display retrieved documents
if show_retrieved_docs and st.session_state.retrieved_docs:
    st.divider()
    with st.expander("📄 Retrieved Documents", expanded=False):
        for i, doc in enumerate(st.session_state.retrieved_docs, 1):
            with st.container(border=True):
                st.markdown(f"**Document {i}:**")
                st.markdown(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

# Display info about what's happening
if st.session_state.messages:
    st.divider()
    with st.expander("ℹ️ RAG Info", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retrieval Method", retrieval_method.split("(")[0].strip())
        with col2:
            st.metric("Documents Retrieved", len(st.session_state.retrieved_docs))
        with col3:
            st.metric("Chat Messages", len(st.session_state.messages))
