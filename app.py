import streamlit as st
import time
import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForQuestionAnswering

# ==========================================
# 1. KONFIGURASI HALAMAN & JUDUL
# ==========================================
st.set_page_config(
    page_title="Chatbot Gizi Pintar",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Asisten Gizi & Kesehatan")
st.markdown("---")
st.markdown("""
Halo! Saya adalah asisten AI yang dilatih khusus untuk menjawab pertanyaan seputar **gizi, diet, dan kesehatan**.
Silakan tanya apa saja, misalnya:
- *"Apa makanan yang baik untuk penderita diabetes?"*
- *"Bagaimana cara menurunkan berat badan secara alami?"*
""")

# ==========================================
# 2. LOAD RESOURCES (CACHED)
# Biar gak loading ulang tiap kali user ngetik
# ==========================================

@st.cache_resource
def init_pinecone():
    """Menghubungkan ke Database Pinecone"""
    # Mengambil API Key dari Secrets (Aman!)
    try:
        api_key = st.secrets["PINECONE_API_KEY"]
    except FileNotFoundError:
        st.error("⚠️ API Key Pinecone belum disetting di Secrets!")
        st.stop()
        
    pc = Pinecone(api_key=api_key)
    index_name = "gizi-knowledge"
    
    # Cek koneksi ke index
    try:
        index = pc.Index(index_name)
        return index
    except Exception as e:
        st.error(f"Gagal konek ke Pinecone: {e}")
        st.stop()

@st.cache_resource
def load_embedding_model():
    """Load model untuk mengubah pertanyaan user jadi vektor"""
    # Model ini harus SAMA dengan yang dipakai saat upload data ke Pinecone
    return SentenceTransformer('LazarusNLP/all-indobert-base-v2')

@st.cache_resource
def load_qa_model():
    """Load otak AI (IndoBERT Fine-Tuned) dari Hugging Face"""
    # GANTI 'username_kamu' dengan username Hugging Face aslimu!
    # Contoh: "budi_santoso/indobert-gizi-qa-final"
    model_name = "username_kamu/indobert-gizi-qa-final" 
    
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForQuestionAnswering.from_pretrained(model_name)
        
        # Gunakan CPU di Streamlit Cloud (Gratis gak dapet GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        return tokenizer, model, device
    except OSError:
        st.error(f"⚠️ Model '{model_name}' tidak ditemukan di Hugging Face. Pastikan nama repo benar dan public.")
        st.stop()

# Load semua komponen
with st.spinner("Sedang menyiapkan otak AI... (Mohon tunggu sebentar)"):
    index = init_pinecone()
    embed_model = load_embedding_model()
    tokenizer, qa_model, device = load_qa_model()

# ==========================================
# 3. FUNGSI LOGIKA (RETRIEVAL & READER)
# ==========================================

def retrieve_documents(query, top_k=3):
    """Langkah 1: Cari dokumen relevan di Pinecone"""
    # Ubah pertanyaan teks -> vektor angka
    query_vector = embed_model.encode(query).tolist()
    
    # Cari di cloud
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True # Wajib True biar teks aslinya kebawa
    )
    
    contexts = []
    for match in results['matches']:
        if 'text' in match['metadata']: # Pastikan ada metadatanya
            contexts.append({
                'text': match['metadata']['text'],
                'title': match['metadata'].get('title', 'Sumber tidak diketahui'),
                'url': match['metadata'].get('url', '#'),
                'score': match['score']
            })
    return contexts

def extract_answer(question, context):
    """Langkah 2: Suruh IndoBERT baca konteks dan cari jawaban"""
    inputs = tokenizer(
        question, 
        context, 
        return_tensors='pt', 
        truncation=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    # Ambil posisi start dan end dengan probabilitas tertinggi
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    
    # Validasi: Jika posisi start > end, berarti model bingung
    if start_idx > end_idx:
        return None, 0.0

    # Hitung confidence score sederhana
    confidence = (torch.max(torch.softmax(outputs.start_logits, dim=1)) * torch.max(torch.softmax(outputs.end_logits, dim=1))).item()

    # Convert token ID kembali ke kata-kata
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
    
    # Bersihkan sisa-sisa token aneh (seperti ##)
    answer = answer.replace('##', '')
    
    return answer, confidence

# ==========================================
# 4. USER INTERFACE (CHAT)
# ==========================================

# Form input agar user bisa tekan Enter
with st.form(key='search_form'):
    query = st.text_input("Tulis pertanyaanmu di sini:", placeholder="Contoh: Apa bahaya makan gorengan?")
    submit_button = st.form_submit_button(label='Tanya Dokter AI 🤖')

if submit_button and query:
    if len(query) < 3:
        st.warning("Pertanyaan terlalu pendek, tolong lengkapi ya.")
    else:
        start_time = time.time()
        
        # 1. RETRIEVAL
        with st.status("🔍 Sedang mencari artikel relevan...", expanded=True) as status:
            st.write("Menghubungi Pinecone...")
            retrieved_docs = retrieve_documents(query)
            
            if not retrieved_docs:
                status.update(label="Gagal", state="error")
                st.error("Maaf, saya tidak menemukan informasi yang relevan di database.")
            else:
                st.write(f"✅ Ditemukan {len(retrieved_docs)} referensi.")
                
                # 2. READER (Cari jawaban terbaik dari dokumen yang ditemukan)
                st.write("📖 Sedang membaca dan menganalisis...")
                
                best_answer = None
                best_score = -1
                best_source = None
                
                # Cek satu per satu dokumen yang didapat
                for doc in retrieved_docs:
                    ans, conf = extract_answer(query, doc['text'])
                    
                    # Logika memilih jawaban terbaik (Confidence harus > 10%)
                    if ans and conf > best_score and conf > 0.1 and "[CLS]" not in ans:
                        best_answer = ans
                        best_score = conf
                        best_source = doc
                
                status.update(label="Selesai!", state="complete", expanded=False)
                
                # 3. TAMPILKAN HASIL
                st.divider()
                
                if best_answer:
                    st.subheader("💡 Jawaban:")
                    st.success(best_answer.capitalize())
                    
                    # Tampilkan Data Pendukung
                    with st.expander("Lihat Sumber & Konteks Asli"):
                        st.markdown(f"**Sumber:** [{best_source['title']}]({best_source['url']})")
                        st.markdown(f"**Relevansi:** {best_score:.2%}")
                        st.info(f"**Kutipan Teks:**\n\n...{best_source['text']}...")
                        
                else:
                    # Fallback jika Retrieval dapat, tapi Reader bingung (skor rendah)
                    st.warning("Saya menemukan artikel yang mungkin relevan, tapi saya kurang yakin jawaban pastinya di bagian mana.")
                    st.markdown("**Coba baca artikel ini langsung:**")
                    for doc in retrieved_docs[:2]:
                         st.markdown(f"- [{doc['title']}]({doc['url']})")

        # Footer waktu eksekusi
        end_time = time.time()
        st.caption(f"Waktu proses: {end_time - start_time:.2f} detik")