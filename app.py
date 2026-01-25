import streamlit as st
import pandas as pd
import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForQuestionAnswering
from rank_bm25 import BM25Okapi
import numpy as np
import time

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Chatbot Gizi & Nutrisi",
    page_icon="🥗",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-link {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #4CAF50;
    }
    .stChatMessage {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD RESOURCES (CACHED)
# ============================================

@st.cache_resource(show_spinner=False)
def load_resources():
    """Load all resources: Pinecone, Embedding Model, BM25"""
    
    # 1. Load Pinecone
    try:
        api_key = st.secrets["PINECONE_API_KEY"]
    except:
        st.error("❌ PINECONE_API_KEY tidak ditemukan di secrets!")
        st.stop()
    
    pc = Pinecone(api_key=api_key)
    index_name = "gizi-knowledge"
    
    try:
        pinecone_index = pc.Index(index_name)
    except Exception as e:
        st.error(f"❌ Gagal koneksi ke Pinecone: {e}")
        st.stop()
    
    # 2. Load Embedding Model
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # 3. Load BM25 data (optional, untuk hybrid search)
    try:
        df = pd.read_csv('processed_chunks.csv')
        tokenized_corpus = [doc.lower().split() for doc in df['text'].tolist()]
        bm25 = BM25Okapi(tokenized_corpus)
    except:
        df = None
        bm25 = None
    
    return pinecone_index, embedding_model, df, bm25

# ============================================
# SEARCH FUNCTIONS
# ============================================

def retrieve_from_pinecone(query, pinecone_index, embedding_model, top_k=3):
    """Retrieve documents from Pinecone"""
    query_vector = embedding_model.encode(query).tolist()
    
    results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    docs = []
    for match in results['matches']:
        if 'text' in match['metadata']:
            docs.append({
                'text': match['metadata']['text'],
                'title': match['metadata'].get('title', 'Tidak diketahui'),
                'url': match['metadata'].get('url', '#'),
                'intent': match['metadata'].get('intent', ''),
                'score': match['score']
            })
    
    return docs

def hybrid_retrieve(query, pinecone_index, embedding_model, bm25, df, top_k=5):
    """Hybrid search: Pinecone + BM25"""
    
    # Dense retrieval (Pinecone)
    dense_docs = retrieve_from_pinecone(query, pinecone_index, embedding_model, top_k * 2)
    
    if bm25 is None or df is None:
        return dense_docs[:top_k]
    
    # BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
    
    # Combine scores
    combined = {}
    
    for doc in dense_docs:
        key = doc['text'][:50]
        combined[key] = {
            'doc': doc,
            'score': 0.6 * doc['score']
        }
    
    for idx in top_bm25_indices:
        key = df.iloc[idx]['text'][:50]
        bm25_score = float(bm25_scores[idx])
        
        if key in combined:
            combined[key]['score'] += 0.4 * bm25_score
        else:
            combined[key] = {
                'doc': {
                    'text': df.iloc[idx]['text'],
                    'title': df.iloc[idx]['title'],
                    'url': df.iloc[idx]['url'],
                    'intent': df.iloc[idx]['intent'],
                    'score': bm25_score
                },
                'score': 0.4 * bm25_score
            }
    
    sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in sorted_results[:top_k]]

def search_snippet_mode(question, pinecone_index, embedding_model, df, bm25, top_k=3):
    """Search engine mode: return snippet"""
    
    # Retrieve documents
    if bm25 is not None:
        retrieved_docs = hybrid_retrieve(question, pinecone_index, embedding_model, bm25, df, top_k)
    else:
        retrieved_docs = retrieve_from_pinecone(question, pinecone_index, embedding_model, top_k)
    
    if not retrieved_docs:
        return {
            'answer': 'Informasi tidak ditemukan dalam basis pengetahuan.',
            'source': None,
            'confidence': 0.0,
            'intent': None,
            'all_results': []
        }
    
    # Get best result
    best_doc = retrieved_docs[0]
    snippet = best_doc['text'][:400] + "..." if len(best_doc['text']) > 400 else best_doc['text']
    
    response = (
        f"📄 **{best_doc['title']}**\n\n"
        f"{snippet}\n\n"
        f"_Untuk informasi lengkap, kunjungi sumber artikel._"
    )
    
    return {
        'answer': response,
        'source': best_doc['url'],
        'confidence': best_doc['score'],
        'intent': best_doc.get('intent', ''),
        'all_results': retrieved_docs
    }

# ============================================
# LOAD RESOURCES
# ============================================

with st.spinner("🔄 Memuat sistem..."):
    pinecone_index, embedding_model, df, bm25 = load_resources()

# ============================================
# UI LAYOUT
# ============================================

# Header
st.markdown('<div class="main-header">🥗 Chatbot Gizi & Nutrisi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asisten Pencarian Informasi Kesehatan, Gizi, Diet, dan Nutrisi</div>', unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================
# DISPLAY CHAT HISTORY (DI ATAS FORM)
# ============================================

if st.session_state.messages:
    st.markdown("### 💬 Riwayat Percakapan")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                
                if meta.get('source'):
                    with st.expander("📎 Sumber & Detail"):
                        st.markdown(f"**Confidence:** `{meta['confidence']:.2%}`")
                        if meta.get('intent'):
                            st.markdown(f"**Topik:** `{meta['intent']}`")
                        st.markdown(f'<div class="source-link">🔗 <a href="{meta["source"]}" target="_blank">{meta["source"]}</a></div>', 
                                  unsafe_allow_html=True)
                        
                        if meta.get('all_results') and len(meta['all_results']) > 1:
                            st.markdown("**📚 Hasil Lainnya:**")
                            for i, doc in enumerate(meta['all_results'][1:], 2):
                                st.caption(f"{i}. {doc['title']} (Score: {doc['score']:.2f})")
    
    st.divider()

# ============================================
# SEARCH FORM (DI BAWAH CHAT HISTORY)
# ============================================

st.markdown("### 🔍 Cari Informasi")

# Example questions as expander
with st.expander("📚 Lihat Contoh Pertanyaan", expanded=False):
    col1, col2 = st.columns(2)
    
    examples = [
        "Kandungan gizi telur",
        "Cara menurunkan berat badan",
        "Penyebab perut buncit",
        "Makanan untuk diet",
        "Penyebab anemia",
        "Makanan untuk ibu hamil",
        "Gejala diabetes",
        "Manfaat vitamin C"
    ]
    
    for i, example in enumerate(examples):
        with col1 if i % 2 == 0 else col2:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                # Set question in session state
                st.session_state['selected_question'] = example
                st.rerun()

# Main search form
with st.form(key='search_form', clear_on_submit=True):
    # Check if there's a selected question from examples
    default_question = st.session_state.pop('selected_question', '')
    
    question = st.text_input(
        "Tulis pertanyaan Anda:",
        value=default_question,
        placeholder="Contoh: Apa makanan yang baik untuk penderita diabetes?"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        submit_button = st.form_submit_button("🔍 Cari", use_container_width=True, type="primary")
    with col2:
        clear_button = st.form_submit_button("🗑️ Hapus", use_container_width=True)

# Handle clear button
if clear_button:
    st.session_state.messages = []
    st.rerun()

# Handle search
if submit_button and question:
    if len(question) < 3:
        st.warning("⚠️ Pertanyaan terlalu pendek. Silakan lengkapi pertanyaan Anda.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Perform search
        with st.spinner("🔍 Mencari informasi relevan..."):
            start_time = time.time()
            
            result = search_snippet_mode(
                question,
                pinecone_index,
                embedding_model,
                df,
                bm25,
                top_k=3
            )
            
            search_time = time.time() - start_time
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "metadata": {
                "confidence": result['confidence'],
                "source": result['source'],
                "intent": result.get('intent'),
                "search_time": search_time,
                "all_results": result['all_results']
            }
        })
        
        # Rerun to show new messages above the form
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em; padding: 1rem;'>
        <strong>Chatbot Gizi & Nutrisi</strong><br>
        Powered by Pinecone · IndoBERT · Sentence Transformers<br>
        ⚠️ <em>Informasi bersifat edukatif. Konsultasi dengan ahli untuk saran personal.</em>
    </div>
    """,
    unsafe_allow_html=True
)