import streamlit as st
import time
import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from rag_engine import UpworkRAGEngine

current_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_directory, '.env')
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("DEEPINFRA_API_KEY")
BASE_URL = os.getenv("DEEPINFRA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_KEY or not BASE_URL or not MODEL_NAME:
    st.error("Execution Stopped: High Priority Credentials Missing! Check your .env parameters.")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(proxy=None) if hasattr(httpx, 'Client') else None
)

if "rag" not in st.session_state:
    rag = UpworkRAGEngine()
    try:
        if not rag.load_local_store():
            docs = rag.ingest_and_sanity_check()
            chunks = rag.chunk_documents(docs)
            rag.build_vector_store(chunks)
        st.session_state.rag = rag
    except Exception as e:
        st.error(f"Fatal System Initialization Failure: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

def query_llm_api(prompt_text):
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.01,  
            max_tokens=600
        )
        return response.choices[0].message.content.strip(), time.time() - start_time
    except Exception as e:
        return f"DeepInfra Connection Failure: {str(e)}", time.time() - start_time

# --- Presentation Layer ---
st.set_page_config(page_title="Upwork API Chatbot Assistant", layout="wide")
st.title("💼 Upwork Technical Support Assistant")
st.caption("Senior Enterprise Consultant AI Agent Engine (DeepInfra-Powered)")

# SIDEBAR ARCHITECTURE: Clean history logs on the left column side
st.sidebar.header("📜 Chat Session History Log")
if st.sidebar.button("🗑️ Reset Chat History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")

for idx, old_msg in enumerate(st.session_state.messages):
    st.sidebar.markdown(f"**Q{idx+1}: {old_msg['query']}**")
    st.sidebar.caption(f"🤖 {old_msg['short_answer']}")
    st.sidebar.markdown("---")

# MAIN PAGE: Input layout clean block up top
user_query = st.text_input("📍 Input developer technical query:", placeholder="Type your question here and press Enter...", key="query_input")

if user_query:
    if not st.session_state.messages or st.session_state.messages[0]["query"] != user_query:
        with st.spinner("Analyzing vector clusters and routing inference pipeline..."):
            
            # Step A: High-Density Context Blocks Extraction
            retrieved_fragments = st.session_state.rag.retrieve_context(user_query, k=5) 
            formatted_context = "\n\n".join([f"[Document Block {idx+1}]:\n{item.page_content}" for idx, item in enumerate(retrieved_fragments)])
            raw_fragments_text = [item.page_content for item in retrieved_fragments]
            
            # Step B: Perfectly Calibrated Prompts allowing text cross-referencing
            structured_system_prompt = (
                "You are an expert technical support engineer for Upwork APIs.\n"
                "Your objective is to answer the developer's question accurately using ONLY the facts explicitly stated within the context fragments below.\n\n"
                "CRITICAL REASONING INSTRUCTIONS:\n"
                "1. Read all context fragments carefully. Scan for words, headers, and numbers that match the query terms.\n"
                "2. When asked about token lifetimes, look specifically for numbers linked to 'access token' or 'refresh token' across the text segments.\n"
                "3. If the context fragments contain headers, descriptions, or variables related to the query, combine those facts into a clean, direct answer.\n"
                "4. If and only if the text fragments are completely silent on the topic, respond exactly with:\n"
                "\"I'm sorry, but the provided documentation does not contain that information.\"\n"
                "5. Never extrapolate or rely on external assumptions.\n\n"
                f"--- PROVIDED CONTEXT FRAGMENTS ---\n{formatted_context}\n\n"
                f"Developer Question: {user_query}\n"
                "Helpful Technical Response:"
            )
            
            final_answer, performance_latency = query_llm_api(structured_system_prompt)
            short_preview = final_answer[:50] + "..." if len(final_answer) > 50 else final_answer
            
            st.session_state.messages.insert(0, {
                "query": user_query,
                "content": final_answer,
                "short_answer": short_preview,
                "latency": performance_latency,
                "fragments": raw_fragments_text
            })

if st.session_state.messages:
    active_response = st.session_state.messages[0]
    st.info(f"❓ **Active Query Context**: {active_response['query']}")
    st.markdown(active_response["content"])
    st.caption(f"⏱ Hopkins Processing Metric (API Latency): {active_response['latency']:.3f} seconds")
    with st.expander("🔍 Verified Context Baselines"):
        for idx, text in enumerate(active_response["fragments"]):
            st.code(text, language="text")