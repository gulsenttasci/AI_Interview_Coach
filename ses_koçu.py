import sounddevice as sd
import numpy as np
import librosa
import speech_recognition
import time


# ---AYARLAR---
FS = 22050  # Örnekleme hızı(librosa standardı)
SURE = 3  # Her 3 saniyede bir analiz yap
THRESHOLD_VOLUME = 0.01  # Sesin kısıklık eşiği
GECERSIZ_KELIMELER = ["eee", "ııı", "şey", "yani", "ımm", "mmm", "aaa"]


def analiz_et():
    r = sr.Recognizer()

    print("Mülakat başladı! Sesin ve duygu durumun analiz ediliyor...")

    while True:
        # 1. SES KAYDI(Anlık veri çekme)
        kayit = sd.rec(int(SURE * FS), samplerate=FS, channels=1)
        sd.wait()
        ses_verisi = kayit.flatten()

        # 2.SES SEVİYESİ(VOLUME) ANALİZİ
        rms = np.sqrt(np.mean(ses_verisi**2))  # Kareli ortalama(Enerji)
        if rms < THRESHOLD_VOLUME:
            print(
                "SESİN ÇOK KISIK: Biraz daha özgüvenli ve yüksek sesle konuşabilirsin."
            )

        # 3.DUYGU VE STRES ANALİZİ(Librosa ile Pitch Takibi)
        # Heyecanlandığında sesin frekansı(pitch) artar ve titrer.
        pitches, magnitudes = librosa.piptrack(y=ses_verisi, sr=FS)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        if pitch > 300:  # Yüksek frekans genellikle stres/heyecan belirtisidir
            print(
                "SAKİN OL: Ses tonunda hafif bir heyecan seziyorum,derin bir nefes al ve gevşe."
            )

        # 4.METİN ANALİZİ(WPM ve "eee,ııı...")
        try:
            # Sesi geçici bir dosyaya yazmadan direkt analiz etmek için Raw kullanabiliriz
            # Şimdilik basitlik için standardı  kullanalım
            with sr.Microphone() as kaynak:
                audio = r.listen(kaynak, phrase_time_limit=SURE)
                metin = r.recognize_google(audio, language="tr-TR").lower()

                # Gereksiz kelime kontrolü
                bulunan_dolgular = [
                    kelime for kelime in GECERSIZ_KELIMELER if kelime in metin
                ]
                if bulunan_dolgular:
                    print(
                        f"DİKKAT: Çok fazla '{bulunan_dolgular[0]}' kullanıyorsun, duraksamaktan çekinme :)"
                    )

                # WPM(Dakikadaki Kelime Sayısı) Hesaplama
                kelime_sayisi = len(metin.split())
                wpm = (kelime_sayisi / SURE) * 60
                if wpm > 150:
                    print(
                        f"ÇOK HIZLI KONUŞUYORSUN({int(wpm)} WPM): Biraz yavaşla.Ne acelen var :)"
                    )

        except:
            pass  # Ses anlaşılmazsa sessizce devam et


# Çalıştır
analiz_et()
