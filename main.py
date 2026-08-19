from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:54312/v1", 
    api_key="test"
)

print("Modele bağlanılıyor...\n")

try:
    # Az önce bulduğumuz tam model ismini kullanıyoruz
    response = client.chat.completions.create(
        model="Phi-3.5-mini-instruct-generic-gpu:2",
        messages=[{"role": "user", "content": "Merhaba, şu an bilgisayarımda tamamen yerel ve internetsiz olarak mı çalışıyorsun? Lütfen kısaca Türkçe cevap ver."}],
        max_tokens=200,
        temperature=0.7
    )
    
    print("\nAsistan:", response.choices[0].message.content)
    
except Exception as e:
    print("\nBir hata oluştu:", e)