import sounddevice as sd
import numpy as np
import librosa
import speech_recognition as sr
import time


# ---AYARLAR---
FS = 22050  # Örnekleme hızı(librosa standardı)
SURE = 3  # Her 3 saniyede bir analiz yap
THRESHOLD_VOLUME = 0.01  # Sesin kısıklık eşiği
GECERSIZ_KELIMELER = ["eee", "ııı", "şey", "yani", "ımm", "mmm", "aaa"]


def sesi_analiz_et():
    """
    Ses dinleme,
    Frekans(stres) analizi,
    Metine çevirme
    Geri dönüt : Mesaj , Puan Güncelleme

    """
    r = sr.Recognizer()
    puan_kir = False  # flag
    final_mesaj = ""

    try:

        # 1. SES KAYDI VE ANALİZ(Librosa & Numpy)
        # 4GB RAM için dostane bir süre
        kayit = sd.rec(int(SURE * FS), samplerate=FS, channels=1)
        sd.wait()
        ses_verisi = kayit.flatten()

        # 2.SES SEVİYESİ(VOLUME) ANALİZİ
        rms = np.sqrt(np.mean(ses_verisi**2))  # Kareli ortalama(Enerji)
        if rms < THRESHOLD_VOLUME:

            return (
                "SESİN COK KISIK: Biraz yuksek sesle konusabilirsin.",
                True,
            )

        # 3.DUYGU VE STRES ANALİZİ(Librosa ile Pitch Takibi)
        # Heyecanlandığında sesin frekansı(pitch) artar ve titrer.
        pitches, magnitudes = librosa.piptrack(y=ses_verisi, sr=FS)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        if pitch > 300:  # Yüksek frekans genellikle stres/heyecan belirtisidir

            return (
                "SAKİN OL: Ses tonunda hafif bir heyecan seziyorum,derin bir nefes al ve gevşe.",
                False,
            )  # Puan kırılmıyor,sadece uyarı

        # 4.METİN ANALİZİ(WPM ve "eee,ııı...(Dolgu Kelimeleri)")

        # Sesi geçici bir dosyaya yazmadan direkt analiz etmek için Raw kullanılabilir
        # Şimdilik basitlik için standardı  kullanalım
        with sr.Microphone() as kaynak:
            # Arka plan hızlıca taranır
            r.adjust_for_ambient_noise(kaynak, duration=0.2)
            audio = r.listen(kaynak, phrase_time_limit=SURE)
            metin = r.recognize_google(audio, language="tr-TR").lower()

            # Gereksiz kelime(Dolgu) kontrolü
            bulunan_dolgular = [
                kelime for kelime in GECERSIZ_KELIMELER if kelime in metin
            ]
            if bulunan_dolgular:
                return f"DIKKAT: Cok fazla '{bulunan_dolgular[0]}' kullaniyorsun.Duraksamaktan çekinme :)", True

            # WPM(Dakikadaki Kelime Sayısı) Hesaplama
            wpm = (len(metin.split()) / SURE) * 60
            if wpm > 150:
                return f"COK HIZLI KONUSUYORSUN ({int(wpm)} WPM): Biraz yavasla.", True

    except Exception as e:
        return "Mülakat başladı,dinliyorum...", False


# Test
if __name__ == "__main__":
    print("Ses modülü test ediliyor...")
    while True:
        print(sesi_analiz_et())
