# 🏄‍♀️ Türkiye Su Sporları Rehberi (Local RAG)

Bu proje, Türkiye'deki su sporları rotalarını (SUP, Windsurf, Kitesurf, Dalış) kapsayan, tamamen yerel (çevrimdışı) çalışan bir RAG (Retrieval-Augmented Generation) asistanıdır.

## 🚀 Teknolojiler
* **Dil Modeli (LLM):** Phi-3.5 Mini (Microsoft Foundry Local üzerinden)
* **Vektör Veritabanı:** ChromaDB
* **Embedding:** paraphrase-multilingual-MiniLM-L12-v2
* **Metin Parçalama:** LangChain (RecursiveCharacterTextSplitter)
* **Arayüz:** Streamlit

## ⚙️ Kurulum
1. Gerekli kütüphaneleri yükleyin: `pip install streamlit chromadb langchain langchain-community sentence-transformers openai`
2. Veritabanını oluşturun: `python ingest.py`
3. Uygulamayı başlatın: `streamlit run app.py`
