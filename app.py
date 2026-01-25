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
def load_resources():
    """Load all resources: Pinecone, Embedding Model, QA Model, BM25"""
    
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
    
    # 3. Load QA Model dari Hugging Face
    model_name = "SonaRFD/indobert-gizi-qa-final"
    
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        qa_model = BertForQuestionAnswering.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qa_model.to(device)
    except Exception as e:
        st.error(f"❌ Gagal load model QA: {e}")
        st.stop()
    
    # 4. Load BM25 data (optional, untuk hybrid search)
    # Jika Anda punya processed_chunks.csv di repo
    try:
        df = pd.read_csv('processed_chunks.csv')
        tokenized_corpus = [doc.lower().split() for doc in df['text'].tolist()]
        bm25 = BM25Okapi(tokenized_corpus)
    except:
        df = None
        bm25 = None
    
    return pinecone_index, embedding_model, tokenizer, qa_model, device, df, bm25

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
        # Use text as key (simplified)
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

def extract_answer_qa(question, context, tokenizer, qa_model, device):
    """Extract answer using QA model"""
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    start_idx = torch.argmax(outputs.start_logits[0]).item()
    end_idx = torch.argmax(outputs.end_logits[0]).item()
    
    start_probs = torch.softmax(outputs.start_logits[0], dim=0)
    end_probs = torch.softmax(outputs.end_logits[0], dim=0)
    confidence = (start_probs[start_idx] * end_probs[end_idx]).item()
    
    if start_idx > end_idx or start_idx == 0:
        return None, confidence
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])
    answer = answer.replace('##', '').strip()
    
    return answer, confidence

def search_pipeline(question, pinecone_index, embedding_model, tokenizer, qa_model, 
                   device, df, bm25, mode='search_engine', threshold=0.1, top_k=3):
    """Main search pipeline"""
    
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
    
    if mode == 'search_engine':
        # Search engine mode: return snippet
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
    
    else:
        # QA mode: extract answer
        best_answer = None
        best_confidence = 0
        best_doc = None
        
        for doc in retrieved_docs:
            answer, conf = extract_answer_qa(question, doc['text'], tokenizer, qa_model, device)
            
            if answer and conf > best_confidence and conf > threshold:
                best_answer = answer
                best_confidence = conf
                best_doc = doc
        
        if best_answer:
            response = f"Berdasarkan artikel \"{best_doc['title']}\".\n\n{best_answer}"
        else:
            response = 'Informasi tidak ditemukan dalam basis pengetahuan.'
            best_doc = None
        
        return {
            'answer': response,
            'source': best_doc['url'] if best_doc else None,
            'confidence': best_confidence,
            'intent': best_doc.get('intent', '') if best_doc else None,
            'all_results': retrieved_docs
        }

# ============================================
# LOAD RESOURCES
# ============================================

with st.spinner("🔄 Memuat sistem..."):
    pinecone_index, embedding_model, tokenizer, qa_model, device, df, bm25 = load_resources()

# ============================================
# UI LAYOUT
# ============================================

# Header
st.markdown('<div class="main-header">🥗 Chatbot Gizi & Nutrisi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asisten Pencarian Informasi Kesehatan, Gizi, Diet, dan Nutrisi</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Mode selection
    search_mode = st.radio(
        "Mode Pencarian",
        options=['search_engine', 'qa_model'],
        format_func=lambda x: "Search Engine (Snippet)" if x == 'search_engine' else "QA Model (Extract Answer)",
        index=0,
        help="Search Engine: Tampilkan snippet. QA Model: Ekstrak jawaban spesifik."
    )
    
    st.divider()
    
    # Parameters
    st.subheader("🎚️ Parameters")
    
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Minimum confidence untuk menampilkan hasil"
    )
    
    top_k = st.slider(
        "Jumlah Hasil",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Info Sistem")
    st.info(f"**Mode:** {search_mode}")
    st.info(f"**Device:** {device.upper()}")
    st.info(f"**Hybrid Search:** {'✅' if bm25 is not None else '❌'}")

# Main content
col_main, col_side = st.columns([2, 1])

with col_main:
    st.header("💬 Pencarian")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
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
    
    # Chat input
    if question := st.chat_input("🔍 Cari informasi gizi, nutrisi, diet, atau kesehatan..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari informasi..."):
                start_time = time.time()
                
                result = search_pipeline(
                    question,
                    pinecone_index,
                    embedding_model,
                    tokenizer,
                    qa_model,
                    device,
                    df,
                    bm25,
                    mode=search_mode,
                    threshold=threshold,
                    top_k=top_k
                )
                
                search_time = time.time() - start_time
                
                st.markdown(result['answer'])
                
                if result['source']:
                    with st.expander("📎 Sumber & Detail", expanded=True):
                        st.markdown(f"**Confidence:** `{result['confidence']:.2%}`")
                        if result.get('intent'):
                            st.markdown(f"**Topik:** `{result['intent']}`")
                        st.markdown(f"**Waktu:** `{search_time:.2f}s`")
                        st.markdown(f'<div class="source-link">🔗 <a href="{result["source"]}" target="_blank">{result["source"]}</a></div>', 
                                  unsafe_allow_html=True)
                        
                        if len(result['all_results']) > 1:
                            st.markdown("**📚 Hasil Lainnya:**")
                            for i, doc in enumerate(result['all_results'][1:], 2):
                                st.caption(f"{i}. {doc['title']} (Score: {doc['score']:.2f})")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "metadata": {
                        "confidence": result['confidence'],
                        "source": result['source'],
                        "intent": result.get('intent'),
                        "all_results": result['all_results']
                    }
                })

with col_side:
    st.header("📚 Contoh Pencarian")
    
    examples = {
        "🍎 Fakta Gizi": [
            "Kandungan gizi telur",
            "Kalori nasi putih",
            "Manfaat vitamin C"
        ],
        "⚖️ Diet": [
            "Cara menurunkan berat badan",
            "Penyebab perut buncit",
            "Makanan untuk diet"
        ],
        "💪 Kesehatan": [
            "Penyebab anemia",
            "Makanan untuk ibu hamil",
            "Gejala diabetes"
        ]
    }
    
    for category, questions in examples.items():
        with st.expander(f"**{category}**"):
            for q in questions:
                if st.button(q, key=q, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    st.divider()
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        <strong>Chatbot Gizi & Nutrisi</strong><br>
        Powered by Pinecone · IndoBERT · Hugging Face<br>
        ⚠️ <em>Informasi bersifat edukatif. Konsultasi dengan ahli untuk saran personal.</em>
    </div>
    """,
    unsafe_allow_html=True
)