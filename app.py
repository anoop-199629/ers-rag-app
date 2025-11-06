import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import anthropic
import json
import hashlib

st.set_page_config(page_title="ERS RAG System", page_icon="📊", layout="wide")

# Add custom styling
st.markdown("""
<style>
    .source-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .document-selector {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

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
    if database_path and __import__('os').path.exists(database_path):
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
    
    if not __import__('os').path.exists("chunks.jsonl"):
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

# CACHE AVAILABLE DOCUMENTS
@st.cache_resource
def get_available_documents():
    """Extract list of unique documents from chunks.jsonl"""
    documents = set()
    
    try:
        with open("chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                source = chunk.get("metadata", {}).get("source", "Unknown")
                documents.add(source)
    except:
        pass
    
    return sorted(list(documents))

# Load database
try:
    collection = build_and_load_database()
except Exception as e:
    st.error(f"Database error: {str(e)}")
    st.stop()

# Get available documents
available_docs = get_available_documents()

# Load API KEY
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("❌ API key not found in Secrets!")
    st.info("Go to Settings → Secrets and add: ANTHROPIC_API_KEY = \"sk-ant-...\"")
    st.stop()

# Initialize client
try:
    client = anthropic.Anthropic(api_key=api_key)
    st.write("✅ API connected!")
except Exception as e:
    st.error(f"API Error: {str(e)}")
    st.stop()

# ========== SIDEBAR: DOCUMENT SELECTOR ==========
with st.sidebar:
    st.header("📚 Available Documents")
    
    if available_docs:
        st.markdown("<div class='document-selector'>", unsafe_allow_html=True)
        st.write(f"**Total Documents:** {len(available_docs)}")
        st.write("**Documents in system:**")
        for doc in available_docs:
            st.write(f"• {doc}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # DOCUMENT FILTER
        st.subheader("🔍 Filter by Document")
        filter_option = st.radio(
            "Search in:",
            ["All Documents"] + available_docs,
            key="doc_filter"
        )
    else:
        st.warning("No documents loaded yet!")
        filter_option = "All Documents"
    
    st.divider()
    
    # INSTRUCTIONS
    st.subheader("💡 How to Use")
    with st.expander("View Instructions"):
        st.markdown("""
        1. **Select a document** using the filter above
        2. **Ask a question** about that document
        3. **See the source** - every answer shows which document it came from
        4. **Get page numbers** - know exactly where the info is found
        
        **Example questions:**
        - What is the Fiduciary Risk Rating?
        - Summarize the fund performance
        - What are the key metrics?
        """)

# ========== MAIN CONTENT ==========
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.subheader("💬 Chat Interface")
    
    # DISPLAY CHAT HISTORY
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            # Display assistant message with sources
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if "sources" in msg:
                    st.markdown("<div class='source-box'>", unsafe_allow_html=True)
                    st.write("**📄 Sources:**")
                    for source in msg["sources"]:
                        st.write(f"• {source['source']} (Page {source['page']})")
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # USER INPUT
    if q := st.chat_input("Ask about the documents..."):
        st.session_state.messages.append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        
        with st.spinner("🔍 Searching documents..."):
            try:
                # SEARCH DATABASE WITH OPTIONAL FILTERING
                if filter_option != "All Documents":
                    # Filter by selected document
                    all_results = collection.query(query_texts=[q], n_results=10)
                    
                    # Filter results to only selected document
                    filtered_docs = []
                    filtered_metas = []
                    
                    for doc, meta in zip(all_results['documents'][0], all_results['metadatas'][0]):
                        if meta.get('source') == filter_option:
                            filtered_docs.append(doc)
                            filtered_metas.append(meta)
                            if len(filtered_docs) >= 3:
                                break
                    
                    if not filtered_docs:
                        answer = f"No relevant information found in '{filter_option}' about your question."
                        sources = []
                    else:
                        results_docs = filtered_docs
                        results_metas = filtered_metas
                        found_results = True
                else:
                    # Search all documents
                    results = collection.query(query_texts=[q], n_results=3)
                    results_docs = results['documents'][0]
                    results_metas = results['metadatas'][0]
                    found_results = True
                
                if found_results and results_docs:
                    # BUILD CONTEXT WITH SOURCE TRACKING
                    context = ""
                    sources = []
                    
                    for i, (doc, meta) in enumerate(zip(results_docs, results_metas), 1):
                        source_name = meta.get('source', 'Unknown')
                        page_num = meta.get('page', 'Unknown')
                        doc_type = meta.get('type', 'text')
                        
                        context += f"\n[Source {i}: {source_name} - Page {page_num} - {doc_type}]\n{doc}\n"
                        sources.append({
                            "source": source_name,
                            "page": page_num,
                            "type": doc_type
                        })
                    
                    # CALL CLAUDE API
                    prompt = f"""Based on the following document excerpts, answer the question accurately and comprehensively.

DOCUMENTS:
{context}

QUESTION: {q}

ANSWER:"""
                    
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    
                    answer = response.content[0].text
                else:
                    answer = "No relevant documents found for your question. Try asking about a different topic or select 'All Documents' to search broadly."
                    sources = []
                
                # DISPLAY ANSWER WITH SOURCES
                with st.chat_message("assistant"):
                    st.write(answer)
                    if sources:
                        st.markdown("<div class='source-box'>", unsafe_allow_html=True)
                        st.write("**📄 Sources Used:**")
                        for source in sources:
                            st.write(f"• **{source['source']}** (Page {source['page']} - {source['type']})")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # STORE IN HISTORY
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.subheader("📊 Statistics")
    
    # Show document stats
    if available_docs:
        st.markdown("**📚 Documents Loaded:**")
        st.metric("Total Documents", len(available_docs))
    
    # Show query stats
    q_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    if q_count > 0:
        st.metric("Questions Asked", q_count)
        cost = q_count * 0.002
        st.metric("Estimated Cost", f"${cost:.4f}")
    else:
        st.info("📝 Ask a question to see statistics!")
    
    st.divider()
    
    # Show recent sources
    if st.session_state.messages:
        st.subheader("🔍 Recent Sources")
        recent_sources = set()
        for msg in st.session_state.messages[-10:]:
            if "sources" in msg:
                for source in msg["sources"]:
                    recent_sources.add(source['source'])
        
        if recent_sources:
            for source in sorted(recent_sources):
                st.write(f"✓ {source}")
        else:
            st.write("No sources yet")

st.divider()
st.markdown("<div style='text-align: center; color: #888; font-size: 0.9rem;'>Built with Streamlit | Powered by Claude API | RAG System</div>", unsafe_allow_html=True)