import streamlit as st
import chromadb
from foundry_local import FoundryLocalManager
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Türkiye Su Sporları Rehberi", page_icon="🏄‍♀️", layout="wide")

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Sistem Bilgisi")
    st.markdown("Bu RAG asistanı **Microsoft Foundry Local** üzerinde tamamen çevrimdışı çalışmaktadır.")
    
    st.divider()
    st.markdown("🧠 **LLM:** Phi-3.5 Mini")
    st.markdown("📐 **Embedding:** paraphrase-multilingual")
    st.markdown("⚡ **Veritabanı:** ChromaDB")
    
    st.divider()
    st.success("🟢 Sistem Çevrimiçi (Yerel)")
    
    st.divider()
    st.markdown("### 👩‍💻 Geliştirici")
    st.markdown("**Aylin Elif Gökdemir**")
    st.markdown("[LinkedIn Profilim](https://www.linkedin.com/in/aylinelifgokdemir/) | [GitHub Profilim](https://github.com/aylinelif)")

# --- ANA EKRAN BÖLÜMÜ ---
st.markdown('<h1 style="color: #1F618D; font-weight: 700; text-align: center;">🏄‍♀️ Türkiye Su Sporları Rehberi</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5D6D7E;'>Yerel yapay zeka ile Türkiye'nin en iyi sörf, dalış ve rafting rotalarını keşfedin.</p>", unsafe_allow_html=True)

# Mimariyi Sergileme Alanı (Expander)
with st.expander("🛠️ Bu Sistem Nasıl Çalışır? (Mimari Özeti)"):
    st.markdown("""
    * **Güvenlik ve Gizlilik:** Verileriniz buluta gitmez, %100 yerel donanımda (Microsoft Foundry Local) çalışır.
    * **Vektör Arama:** Sorularınız anlık olarak *ChromaDB* üzerinde taranır ve en alakalı bilgiler getirilir.
    * **Halüsinasyonsuz Üretim:** *Phi-3.5 Mini* modeli, sadece bu doğrulanmış kaynakları kullanarak size doğal bir dille cevap verir.
    """)

# --- SİSTEMİ HAZIRLAMA ---
@st.cache_resource
def sistemi_hazirla():
    manager = FoundryLocalManager()
    manager.start_service()
    
    alias = "phi-3.5-mini"
    manager.download_model(alias)
    manager.load_model(alias)
    gercek_model_id = manager.get_model_info(alias).id
    
    local_client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="su_sporlari")
    
    return local_client, embed_model, gercek_model_id, collection

client, embedder, CHAT_MODEL_ID, collection = sistemi_hazirla()

# --- KULLANICI ETKİLEŞİMİ VE SOHBET ---
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = [
        {"rol": "assistant", "icerik": "Merhaba! 🏄‍♀️ Türkiye'nin dört bir yanındaki su sporları rotaları hakkında bana her şeyi sorabilirsiniz. İster hızlı butonları kullanın, isterseniz de sorunuzu aşağıya yazın!"}
    ]

for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["icerik"])

# Örnek Soru Butonları (UX Geliştirmesi)
st.markdown("💡 **Hızlı Sorular:**")
kolon1, kolon2, kolon3 = st.columns(3)

if "ornek_soru" not in st.session_state:
    st.session_state.ornek_soru = ""

if kolon1.button("Rüzgar sörfü için en iyi yerler neresi?"):
    st.session_state.ornek_soru = "Rüzgar sörfü için en iyi yerler neresi?"
if kolon2.button("Fethiye çevresinde dalış noktaları?"):
    st.session_state.ornek_soru = "Fethiye çevresinde dalış noktaları?"
if kolon3.button("Kitesurf'e yeni başlayacağım, tavsiye ver."):
    st.session_state.ornek_soru = "Kitesurf'e yeni başlayacağım, tavsiye ver."

# Kullanıcıdan gelen asıl input
soru = st.chat_input("Başka bir şey sor... (Örn: Köprülü Kanyonda ne yapılır?)")

# Hızlı butonlardan veya chat inputtan gelen veriyi birleştirme
aktif_soru = soru if soru else st.session_state.ornek_soru

if aktif_soru:
    # Hızlı butona tıklandıktan sonra döngüyü sıfırlıyoruz
    st.session_state.ornek_soru = ""
    
    st.session_state.mesajlar.append({"rol": "user", "icerik": aktif_soru})
    with st.chat_message("user"):
        st.markdown(aktif_soru)

    with st.chat_message("assistant"):
        with st.spinner("Rehber taranıyor..."):
            soru_vektoru = embedder.encode(aktif_soru).tolist()
            sonuclar = collection.query(query_embeddings=[soru_vektoru], n_results=2)
            en_iyi_2_bilgi = "\n\n".join(sonuclar['documents'][0])

        kombine_soru = f"""Aşağıdaki rehber bilgilerini kullanarak kullanıcıya yardımcı ol ve doğal bir dille yanıt ver. 
(Not: SUP, Paddleboard ve Kürek Sörfü aynı anlama gelmektedir).

Rehber Bilgisi:
{en_iyi_2_bilgi}

Soru: {aktif_soru}"""

        gonderilecek_mesajlar = [
            {"role": "system", "content": "Sen uzman bir su sporları asistanısın. Yanıtlarını verirken önemli yerleri **kalın** yaz ve okunabilirliği artırmak için madde işaretleri kullan."},
            {"role": "user", "content": kombine_soru}
        ]

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL_ID,
                messages=gonderilecek_mesajlar,
                stream=True,
                max_tokens=1024
            )
            
            cevap = st.write_stream(response)
            st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})
            
            with st.expander("📚 Kaynaklar (Sistem bu bilgiyi nereden buldu?)"):
                st.info(en_iyi_2_bilgi)
                
        except Exception as e:
            st.error(f"Hata: {str(e)}")
            if len(st.session_state.mesajlar) > 0:
                st.session_state.mesajlar.pop()