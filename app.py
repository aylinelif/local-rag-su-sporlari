import streamlit as st
import chromadb
from foundry_local import FoundryLocalManager
from openai import OpenAI
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Türkiye Su Sporları Rehberi", page_icon="🏄‍♀️", layout="wide")

with st.sidebar:
    st.title("⚙️ Sistem Bilgisi")
    st.markdown("Bu RAG asistanı **Microsoft Foundry Local** üzerinde çalışmaktadır.")
    st.divider()
    st.markdown("🧠 **LLM:** Phi-3.5 Mini")
    st.markdown("📐 **Embedding:** paraphrase-multilingual")
    st.markdown("⚡ **Veritabanı:** ChromaDB")

st.markdown('<h1 style="color: #1F618D; font-weight: 700;">🏄‍♀️ Türkiye Su Sporları Rehberi</h1>', unsafe_allow_html=True)

@st.cache_resource
def sistemi_hazirla():
    manager = FoundryLocalManager()
    manager.start_service()
    
    alias = "phi-3.5-mini"
    manager.download_model(alias)
    manager.load_model(alias)
    gercek_model_id = manager.get_model_info(alias).id
    
    local_client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    # Yeni, daha zeki dil modelimiz
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # ChromaDB Bağlantısı
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="su_sporlari")
    
    return local_client, embed_model, gercek_model_id, collection

client, embedder, CHAT_MODEL_ID, collection = sistemi_hazirla()

# 1. DÜZELTME: İlk açılışta profesyonel bir karşılama mesajı
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = [
        {"rol": "assistant", "icerik": "Merhaba! 🏄‍♀️ Türkiye'nin dört bir yanındaki su sporları rotaları (Sörf, Dalış, Rafting vb.) hakkında bana her şeyi sorabilirsiniz. Nereden başlayalım?"}
    ]

for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["icerik"])

soru = st.chat_input("Örn: Rüzgar sörfüne yeni başlayanlar için nereyi önerirsin?")

if soru:
    st.session_state.mesajlar.append({"rol": "user", "icerik": soru})
    with st.chat_message("user"):
        st.markdown(soru)

    with st.chat_message("assistant"):
        with st.spinner("Rehber taranıyor..."):
            # ChromaDB ile Çok Hızlı Benzerlik Araması
            soru_vektoru = embedder.encode(soru).tolist()
            sonuclar = collection.query(query_embeddings=[soru_vektoru], n_results=2)
            en_iyi_2_bilgi = "\n\n".join(sonuclar['documents'][0])

        kombine_soru = f"""Aşağıdaki rehber bilgilerini kullanarak kullanıcıya yardımcı ol ve doğal bir dille yanıt ver. 
(Not: SUP, Paddleboard ve Kürek Sörfü aynı anlama gelmektedir).

Rehber Bilgisi:
{en_iyi_2_bilgi}

Soru: {soru}"""

        # 2. DÜZELTME: Daha zengin bir görsel format (kalın yazı ve maddeler) için Prompt güncellendi
        gonderilecek_mesajlar = [
            {"role": "system", "content": "Sen uzman bir su sporları asistanısın. Yanıtlarını verirken önemli yerleri **kalın** yaz ve okunabilirliği artırmak için madde işaretleri kullan."},
            {"role": "user", "content": kombine_soru}
        ]

        try:
            # 3. DÜZELTME: Cümlelerin yarım kalmaması için max_tokens limiti eklendi
            response = client.chat.completions.create(
                model=CHAT_MODEL_ID,
                messages=gonderilecek_mesajlar,
                stream=True,
                max_tokens=1024
            )
            
            cevap = st.write_stream(response)
            st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})
            
            with st.expander("📚 Kaynaklar"):
                st.info(en_iyi_2_bilgi)
                
        except Exception as e:
            st.error(f"Hata: {str(e)}")
            if len(st.session_state.mesajlar) > 0:
                st.session_state.mesajlar.pop()