import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import anthropic
import os

# PAGE SETUP
st.set_page_config(
    page_title="ERS RAG System",
    page_icon="📊",
    layout="wide"
)

# TITLE
st.title("📊 ERS Document Intelligence System")
st.markdown("Ask questions about ERS documents powered by Claude AI")

# SIDEBAR
with st.sidebar:
    st.header("💡 Example Questions")
    st.write("""
    - What is the Fiduciary Risk Rating?
    - How did Market Risk Navigator perform?
    - What is the Price Risk Indicator?
    - Compare FRR vs Market Risk Navigator
    """)
    
    st.divider()
    
    st.header("ℹ️ About This System")
    st.write("""
    This system uses Retrieval-Augmented Generation (RAG) to answer questions about ERS documents.
    
    - Technology: Claude AI + Vector Database
    - Cost per question: ~$0.002
    - Speed: 2-5 seconds per answer
    """)

# INITIALIZE SESSION STATE
if 'messages' not in st.session_state:
    st.session_state.messages = []

# CACHE THE EMBEDDING FUNCTION - USE DEFAULT EMBEDDINGS
@st.cache_resource
def get_embedding_function():
    # Use DEFAULT embeddings (no external downloads, no issues!)
    return embedding_functions.DefaultEmbeddingFunction()

# LOAD VECTOR DATABASE WITH CORRECT EMBEDDING
try:
    database_path = os.path.join(os.getcwd(), "chroma_data")
    
    if not os.path.exists(database_path):
        st.error("Database not found at: " + database_path)
        st.info("Make sure chroma_data folder is in the same directory as app.py")
        st.stop()
    
    # Get embedding function (cached)
    embed = get_embedding_function()
    
    # Load database
    chroma_client = chromadb.PersistentClient(path=database_path)
    collection = chroma_client.get_collection(
        name="ers_documents",
        embedding_function=embed
    )
    
    st.write("✅ Database connected successfully!")
    
except Exception as e:
    st.error("Error loading database: " + str(e))
    st.stop()

# LOAD API KEY
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not set!")
    st.info("Set ANTHROPIC_API_KEY environment variable")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# MAIN CONTENT
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # DISPLAY CHAT HISTORY
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # USER INPUT
    user_question = st.chat_input("Ask a question about ERS documents...")
    
    if user_question:
        # ADD USER MESSAGE TO HISTORY
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        
        with st.spinner("Searching documents..."):
            try:
                # SEARCH DATABASE
                results = collection.query(
                    query_texts=[user_question],
                    n_results=3
                )
                
                if not results['documents'] or not results['documents'][0]:
                    answer = "No relevant documents found for your question."
                    sources = []
                else:
                    # BUILD CONTEXT FROM SEARCH RESULTS
                    context = ""
                    sources = []
                    
                    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                        source_name = meta.get('source', 'Unknown')
                        page_num = meta.get('page', 'Unknown')
                        
                        context += "\n[Source " + str(i) + ": " + source_name + " - Page " + str(page_num) + "]\n" + doc + "\n"
                        sources.append({
                            "source": source_name,
                            "page": page_num
                        })
                    
                    # CALL CLAUDE API
                    prompt = "Based on these documents, answer the question accurately and comprehensively. Include specific numbers and findings from the documents. If the answer is not in the documents, say clearly that it's not available.\n\nDOCUMENTS:\n" + context + "\n\nQUESTION: " + user_question + "\n\nANSWER:"
                    
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    answer = response.content[0].text
                
                # DISPLAY ANSWER
                st.chat_message("assistant").write(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                st.error("Error: " + str(e))

with col2:
    st.subheader("📖 Sources Referenced")
    
    if 'sources' in locals() and sources:
        st.write("This response used " + str(len(sources)) + " sources:")
        for i, source in enumerate(sources, 1):
            st.write(str(i) + ". " + source['source'])
            st.write("   Page " + str(source['page']))
    else:
        st.write("Sources will appear here after your first question.")
    
    st.divider()
    
    st.subheader("📊 Statistics")
    num_messages = len(st.session_state.messages)
    num_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
    
    if num_questions > 0:
        st.metric("Questions Asked", num_questions)
        estimated_cost = num_questions * 0.002
        st.metric("Estimated Cost", "$" + str(round(estimated_cost, 4)))
    else:
        st.info("Ask a question to see statistics!")

# FOOTER
st.divider()
st.markdown("<div style='text-align: center; color: #888; font-size: 0.9rem;'>Built with Streamlit | Powered by Claude API | RAG System</div>", unsafe_allow_html=True)