import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print("1. Çok Dilli Embedding Modeli Yüklenecek...")
# Türkçe'yi çok daha iyi anlayan daha güçlü bir model
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("2. Metinler Akıllı Parçalara Bölünüyor...")
with open('docs/kıyı_rotalari.txt', 'r', encoding='utf-8') as f:
    icerik = f.read()

# LangChain ile cümle bütünlüğünü bozmadan 500 karakterlik parçalara ayırıyoruz
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=["\n\n", "\n", ".", " "]
)
parcalar = text_splitter.split_text(icerik)
print(f"Toplam {len(parcalar)} adet akıllı bilgi parçası oluşturuldu.")

print("3. ChromaDB Vektör Veritabanı Hazırlanıyor...")
# Veritabanını klasör olarak projeye kaydedecek
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Eski veriler varsa temizleyip baştan kuruyoruz
try:
    chroma_client.delete_collection(name="su_sporlari")
except:
    pass
collection = chroma_client.create_collection(name="su_sporlari")

print("4. Veriler Veritabanına Gömülüyor...")
vektorler = embed_model.encode(parcalar).tolist()
id_listesi = [f"id_{i}" for i in range(len(parcalar))]

collection.add(
    documents=parcalar,
    embeddings=vektorler,
    ids=id_listesi
)

print("Başarılı! Yeni nesil RAG mimarisi hazır.")