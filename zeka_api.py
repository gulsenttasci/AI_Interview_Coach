import requests


API_KEY = "AIzaSyBUI7_HwggjomXfGsvlNl1a8JQUtZDR2CM"

def cevabi_degerlendir(adayin_cevabi):
    """
    Kullanıcının cevabını doğrudan HTTP isteği (REST API) ile 
    Gemini sunucularına gönderir ve sonucu alır. 
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    
    # Sunucuya gönderilecek verinin formatı (JSON)
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Prompt
    prompt = f"""
    Sen zorlu ama destekleyici bir İnsan Kaynakları uzmanısın. 
    Karşında bir Bilgisayar Mühendisliği öğrencisi var. 
    Mülakatta ona bir soru soruldu ve şu cevabı verdi: '{adayin_cevabi}'
    
    Lütfen bu cevabı en fazla 2 kısa cümleyle değerlendir. 
    Doğrudan adaya (Sen diye) hitap et. Teknik olarak yeterli mi, neyi eklemeliydin söyle.
    """
    
    # Gönderilecek veri paketi
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        # Sunucuya post isteği
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Hata varsa yakala
        
        # Gelen JSON verisinin içinden sadece cevabı (text) çek
        sonuc = response.json()
        degerlendirme_metni = sonuc['candidates'][0]['content']['parts'][0]['text']
        
        return degerlendirme_metni.strip()
        
    except Exception as e:
        print(f"API Hatası Detayı: {e}")
        return "İnternet bağlantımda bir sorun var sanırım, cevabını değerlendiremedim."

# Test Bloğu
if __name__ == "__main__":
    print("REST API Testi yapılıyor...")
    test_cevabi = "Ben projelerimde genellikle Python ve OpenCV kullanıyorum, son projemde bir yüz tanıma sistemi yaptım."
    print("Asistanın Yorumu:")
    print(cevabi_degerlendir(test_cevabi))