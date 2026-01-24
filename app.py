import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import time

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Chatbot Gizi & Nutrisi",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-link {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD RESOURCES (CACHED)
# ============================================

@st.cache_resource(show_spinner=False)
def load_retriever():
    """Load retriever components (cached)"""
    
    # Load dataset
    df = pd.read_csv('output_dataset/processed_chunks.csv')
    
    # ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
    
    try:
        collection = chroma_client.get_collection(
            name="gizi_knowledge",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
        )
    except:
        st.error("❌ ChromaDB collection tidak ditemukan! Jalankan Tahap 5 terlebih dahulu.")
        st.stop()
    
    # BM25
    tokenized_corpus = [doc.lower().split() for doc in df['text'].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return df, collection, bm25

# ============================================
# SEARCH FUNCTIONS
# ============================================

def hybrid_retrieve(query, collection, bm25, df, top_k=5, dense_weight=0.6, bm25_weight=0.4):
    """Hybrid retrieval with configurable weights"""
    
    # Dense retrieval
    dense_results = collection.query(query_texts=[query], n_results=top_k * 2)
    
    dense_docs = []
    for i, doc_id in enumerate(dense_results['ids'][0]):
        idx = int(doc_id.split('_')[1])
        dense_docs.append({
            'idx': idx,
            'text': dense_results['documents'][0][i],
            'metadata': dense_results['metadatas'][0][i],
            'dense_score': 1 / (1 + dense_results['distances'][0][i])
        })
    
    # BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
    bm25_docs = {int(idx): float(bm25_scores[idx]) for idx in top_bm25_indices}
    
    # Combine scores
    combined = {}
    for doc in dense_docs:
        idx = doc['idx']
        combined[idx] = {
            'text': doc['text'],
            'metadata': doc['metadata'],
            'score': dense_weight * doc['dense_score']
        }
    
    for idx, score in bm25_docs.items():
        if idx in combined:
            combined[idx]['score'] += bm25_weight * score
        else:
            combined[idx] = {
                'text': df.iloc[idx]['text'],
                'metadata': {
                    'title': df.iloc[idx]['title'],
                    'url': df.iloc[idx]['url'],
                    'intent': df.iloc[idx]['intent']
                },
                'score': bm25_weight * score
            }
    
    # Sort by score
    sorted_results = sorted(combined.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return [
        {
            'text': item[1]['text'],
            'metadata': item[1]['metadata'],
            'score': item[1]['score']
        }
        for item in sorted_results[:top_k]
    ]

def search_engine_pipeline(question, df, collection, bm25, retriever_threshold=0.1, 
                          snippet_length=400, top_k=3, dense_weight=0.6, bm25_weight=0.4):
    """Search engine pipeline with configurable parameters"""
    
    retrieved_docs = hybrid_retrieve(question, collection, bm25, df, top_k, dense_weight, bm25_weight)
    
    if not retrieved_docs or retrieved_docs[0]['score'] < retriever_threshold:
        return {
            'answer': 'Informasi tidak ditemukan dalam basis pengetahuan. Silakan coba pertanyaan lain atau gunakan kata kunci yang berbeda.',
            'source': None,
            'confidence': 0.0,
            'intent': None,
            'all_results': []
        }
    
    best_doc = retrieved_docs[0]
    full_text = best_doc['text']
    snippet = full_text[:snippet_length] + "..." if len(full_text) > snippet_length else full_text
    
    response = (
        f"📄 **{best_doc['metadata']['title']}**\n\n"
        f"{snippet}\n\n"
        f"_Untuk informasi lengkap, kunjungi sumber artikel di bawah._"
    )
    
    return {
        'answer': response,
        'source': best_doc['metadata']['url'],
        'confidence': best_doc['score'],
        'intent': best_doc['metadata']['intent'],
        'all_results': retrieved_docs
    }

# ============================================
# LOAD RESOURCES
# ============================================

with st.spinner("🔄 Memuat sistem..."):
    df, collection, bm25 = load_retriever()

# ============================================
# UI LAYOUT
# ============================================

# Header
st.markdown('<div class="main-header">🥗 Chatbot Gizi & Nutrisi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asisten Pencarian Informasi Kesehatan, Gizi, Diet, dan Nutrisi</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Advanced settings toggle
    show_advanced = st.checkbox("Tampilkan Pengaturan Lanjutan", value=False)
    
    st.subheader("🎚️ Threshold & Parameters")
    
    retriever_threshold = st.slider(
        "Minimum Relevance Score",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.05,
        help="Score minimum untuk menampilkan hasil"
    )
    
    snippet_length = st.slider(
        "Panjang Snippet (karakter)",
        min_value=200,
        max_value=800,
        value=400,
        step=50,
        help="Jumlah karakter yang ditampilkan dari dokumen"
    )
    
    top_k = st.slider(
        "Jumlah Hasil Pencarian",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Jumlah dokumen yang akan dicari"
    )
    
    if show_advanced:
        st.subheader("🔧 Advanced Settings")
        
        dense_weight = st.slider(
            "Dense Search Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Bobot untuk semantic search (embedding)"
        )
        
        bm25_weight = 1.0 - dense_weight
        st.info(f"BM25 Weight: {bm25_weight:.2f}")
    else:
        dense_weight = 0.6
        bm25_weight = 0.4
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistik Database")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Artikel", len(df['doc_id'].unique()))
        st.metric("Chunks", len(df))
    with col2:
        st.metric("DB Size", f"{collection.count()}")
        st.metric("Topics", len(df['intent'].unique()))

# Main content
col_main, col_side = st.columns([2, 1])

with col_main:
    st.header("💬 Pencarian")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'search_count' not in st.session_state:
        st.session_state.search_count = 0
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message and message["metadata"]:
                meta = message["metadata"]
                
                # Source info
                if meta.get('source'):
                    with st.expander("📎 Sumber & Informasi Tambahan", expanded=False):
                        st.markdown(f"**Relevance Score:** `{meta['confidence']:.2f}`")
                        st.markdown(f"**Topik:** `{meta['intent']}`")
                        st.markdown(f'<div class="source-link">🔗 <a href="{meta["source"]}" target="_blank">{meta["source"]}</a></div>', 
                                  unsafe_allow_html=True)
                        
                        # Alternative results
                        if meta.get('all_results') and len(meta['all_results']) > 1:
                            st.markdown("**📚 Hasil Pencarian Lainnya:**")
                            for i, doc in enumerate(meta['all_results'][1:], 2):
                                with st.container():
                                    st.markdown(f"**{i}. {doc['metadata']['title']}**")
                                    st.caption(f"Score: {doc['score']:.2f} | {doc['text'][:120]}...")
    
    # Chat input
    if question := st.chat_input("🔍 Cari informasi tentang gizi, nutrisi, diet, atau kesehatan..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.search_count += 1
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari informasi relevan..."):
                start_time = time.time()
                
                result = search_engine_pipeline(
                    question, 
                    df, 
                    collection, 
                    bm25,
                    retriever_threshold=retriever_threshold,
                    snippet_length=snippet_length,
                    top_k=top_k,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight
                )
                
                search_time = time.time() - start_time
                
                st.markdown(result['answer'])
                
                # Show metadata
                if result['source']:
                    with st.expander("📎 Sumber & Informasi Tambahan", expanded=True):
                        st.markdown(f"**Relevance Score:** `{result['confidence']:.2f}`")
                        st.markdown(f"**Topik:** `{result['intent']}`")
                        st.markdown(f"**Waktu Pencarian:** `{search_time:.2f}s`")
                        st.markdown(f'<div class="source-link">🔗 <a href="{result["source"]}" target="_blank">{result["source"]}</a></div>', 
                                  unsafe_allow_html=True)
                        
                        # Alternative results
                        if len(result['all_results']) > 1:
                            st.markdown("**📚 Hasil Pencarian Lainnya:**")
                            for i, doc in enumerate(result['all_results'][1:], 2):
                                with st.container():
                                    st.markdown(f"**{i}. {doc['metadata']['title']}**")
                                    st.caption(f"Score: {doc['score']:.2f} | {doc['text'][:120]}...")
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "metadata": {
                        "confidence": result['confidence'],
                        "source": result['source'],
                        "intent": result['intent'],
                        "all_results": result['all_results']
                    }
                })

with col_side:
    st.header("📚 Contoh Pencarian")
    
    example_questions = {
        "🍎 Fakta Gizi": [
            "Kandungan gizi telur",
            "Kalori nasi putih",
            "Manfaat vitamin C",
            "Protein dalam daging ayam"
        ],
        "⚖️ Diet & Berat Badan": [
            "Cara menurunkan berat badan",
            "Penyebab perut buncit",
            "Makanan untuk diet",
            "Diet rendah kalori"
        ],
        "💪 Kesehatan": [
            "Penyebab anemia",
            "Cara meningkatkan imun",
            "Makanan untuk ibu hamil",
            "Gejala diabetes"
        ],
        "🍽️ Resep & Makanan": [
            "Resep sarapan sehat",
            "Buah tinggi antioksidan",
            "Sayuran tinggi serat",
            "Menu makanan sehat"
        ]
    }
    
    for category, questions in example_questions.items():
        with st.expander(f"**{category}**"):
            for q in questions:
                if st.button(q, key=q, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Session info
    st.divider()
    st.markdown("### 📊 Sesi Anda")
    st.info(f"**Total Pencarian:** {st.session_state.search_count}")
    
    # Clear chat button
    if st.button("🗑️ Hapus Riwayat Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.search_count = 0
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em; padding: 1rem;'>
        <strong>Chatbot Gizi & Nutrisi</strong> | Search Engine Mode<br>
        Powered by <code>IndoBERT</code> · <code>ChromaDB</code> · <code>BM25</code><br>
        <br>
        ⚠️ <em>Informasi yang disediakan bersifat edukatif. 
        Konsultasikan dengan ahli gizi profesional untuk saran kesehatan personal.</em>
    </div>
    """,
    unsafe_allow_html=True
)