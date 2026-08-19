import sqlite3
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

def benzerlik_hesapla(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Sistem hazırlanıyor... (Veritabanı ve Modeller yükleniyor)")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_client = OpenAI(base_url="http://127.0.0.1:54312/v1", api_key="test")

conn = sqlite3.connect('gezi_veritabani.db')
c = conn.cursor()

soru = input("\nRehbere bir soru sorun: ")

# 2. ARAMA (RETRIEVAL) - GELİŞTİRİLMİŞ VERSİYON
soru_vektoru = embed_model.encode(soru)
c.execute("SELECT metin, vektor FROM belgeler")
kayitlar = c.fetchall()

# Bütün puanları ve metinleri bir listede toplayacağız
skorlar_ve_metinler = []

for metin, vektor_json in kayitlar:
    kayit_vektoru = np.array(json.loads(vektor_json))
    skor = benzerlik_hesapla(soru_vektoru, kayit_vektoru)
    skorlar_ve_metinler.append((skor, metin))

# Puanlara göre büyükten küçüğe sıralıyoruz
skorlar_ve_metinler.sort(key=lambda x: x[0], reverse=True)

# SADECE EN İYİ 1 DEĞİL, EN İYİ 2 PARÇAYI BİRLEŞTİRİYORUZ
en_iyi_2_bilgi = skorlar_ve_metinler[0][1] + "\n\n" + skorlar_ve_metinler[1][1]

print("\n--- ARKA PLAN (Sistemin Bulduğu Bilgiler) ---")
print(en_iyi_2_bilgi)
print("------------------------------------------\n")

# 3. ÜRETİM (GENERATION)
prompt = f"""Sen uzman bir Ege bölgesi su sporları ve gezi rehberisin. 
Kullanıcıya SADECE aşağıdaki 'Bağlam' kısmında verilen bilgileri kullanarak cevap ver. 
Eğer sorunun cevabı bağlamda yoksa "Bu konuda bir bilgim yok" de, asla kendi kendine bilgi uydurma.

Bağlam: {en_iyi_2_bilgi}

Kullanıcının Sorusu: {soru}"""

print("Asistan düşünüyor...\n")

response = llm_client.chat.completions.create(
    model="Phi-3.5-mini-instruct-generic-gpu:2",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300,
    temperature=0.3 
)

print("ASİSTAN:", response.choices[0].message.content)

conn.close()