import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import anthropic
import os
import json
import hashlib
import itertools

st.set_page_config(page_title="ERS RAG System", page_icon="📊", layout="wide")
st.title("📊 ERS Document Intelligence System")
st.markdown("Ask questions about ERS documents powered by Claude AI")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# CACHE COLLECTION - BUILD ONCE
@st.cache_resource
def build_and_load_database():
    """Build database from chunks.jsonl on first run"""
    
    database_path = "./chroma_data"
    collection_name = "ers_documents"
    
    # Check if already built
    if os.path.exists(database_path):
        st.write("📦 Loading existing database...")
        try:
            embed = embedding_functions.DefaultEmbeddingFunction()
            client = chromadb.PersistentClient(path=database_path)
            collection = client.get_collection(name=collection_name, embedding_function=embed)
            st.write("✅ Database loaded!")
            return collection
        except:
            st.write("Rebuilding database...")
    
    # Build from chunks.jsonl
    st.write("🔨 Building database from chunks...")
    
    if not os.path.exists("chunks.jsonl"):
        st.error("chunks.jsonl not found!")
        st.stop()
    
    # Create database
    embed = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Read chunks.jsonl and add to database
    total_chunks = 0
    batch_ids = []
    batch_docs = []
    batch_metas = []
    batch_size = 50
    
    with open("chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            chunk = json.loads(line)
            content = chunk.get("content", "")
            meta = chunk.get("metadata", {})
            
            # Create ID
            src = str(meta.get("source", ""))
            page = str(meta.get("page", ""))
            typ = str(meta.get("type", ""))
            h = hashlib.sha1(content[:400].encode("utf-8")).hexdigest()
            chunk_id = f"{src}|p{page}|{typ}|{h}"
            
            batch_ids.append(chunk_id)
            batch_docs.append(content)
            batch_metas.append(meta)
            
            total_chunks += 1
            
            # Add in batches
            if len(batch_ids) >= batch_size:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                batch_ids = []
                batch_docs = []
                batch_metas = []
    
    # Add remaining
    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
    
    st.write(f"✅ Database ready! Added {total_chunks} chunks")
    return collection

# Load database
try:
    collection = build_and_load_database()
except Exception as e:
    st.error(f"Database error: {str(e)}")
    st.stop()

# Load API
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not set!")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# Chat
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if q := st.chat_input("Ask..."):
        st.session_state.messages.append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        
        with st.spinner("Searching..."):
            try:
                results = collection.query(query_texts=[q], n_results=3)
                
                if results['documents'][0]:
                    context = "\n".join(results['documents'][0])
                    prompt = f"Answer based on: {context}\n\nQ: {q}"
                    
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    answer = response.content[0].text
                else:
                    answer = "No results found."
                
                st.chat_message("assistant").write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.subheader("📊 Stats")
    q_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    if q_count > 0:
        st.metric("Questions", q_count)
        st.metric("Cost", f"${q_count * 0.002:.4f}")