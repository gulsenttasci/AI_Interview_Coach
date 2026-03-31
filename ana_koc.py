import cv2
import mediapipe as mp
import threading
import time
import numpy as np
import speech_recognition as sr

# --- GLOBAL DEĞİŞKENLER (Thread'ler arası iletişim için) ---
# Bilgisayar mühendisliğinde buna 'Shared Memory' benzeri bir yaklaşım diyoruz.
mesaj_yuz = "Baslatiliyor..."
mesaj_ses = "Dinleniyor..."
puan = 100
calisiyor = True
kare_goruntu = None  # Kameradan gelen görüntünün saklandığı kısım


# --- 1. MODÜL: YÜZ TAKİBİ THREAD'İ ---
def yuz_takibi_islem():
    global mesaj_yuz, puan, calisiyor, kare_goruntu  # kare_goruntu eklendi

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,  # mediapipe default value=0.5
    )
    cap = cv2.VideoCapture(0)

    while calisiyor:
        success, image = cap.read()
        if not success:
            continue

        # RAM dostu olması için görüntüyü küçültelim fakat gösterirken net kalmalı
        image_small = cv2.resize(image, (480, 360))
        rgb_image = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            # Buraya yuz_takip.py'deki o meşhur derinlik ve bakış hesaplarını ekleyeceğiz
            # Şimdilik prototip olarak çalışıyor diyelim
            mesaj_yuz = "Kameraya Bakiyor"
        else:
            mesaj_yuz = "YUZ BULUNAMADI!"
            puan -= 0.01  # Hafif puan kırışı
        # Görüntüyü ana döngüye gönder (Kritik Adım)
        kare_goruntu = image
        # CPU'yu %100 yapmamak için çok kısa bir mola
        time.sleep(0.01)

    cap.release()


# --- 2. MODÜL: SES ANALİZİ THREAD'İ ---
def ses_analizi_islem():
    global mesaj_ses, puan, calisiyor
    r = sr.Recognizer()

    while calisiyor:
        with sr.Microphone() as source:
            try:
                # 2 saniyelik kısa dinlemeler RAM'i korur
                audio = r.listen(source, phrase_time_limit=2)
                text = r.recognize_google(audio, language="tr-TR").lower()

                if "eee" in text or "ııı" in text:
                    mesaj_ses = "Dolgulari Azalt!"
                    puan -= 2
                else:
                    mesaj_ses = f"Denen: {text[:15]}..."
            except:
                mesaj_ses = "Gurultu veya Sessizlik"

        time.sleep(0.5)  # Ses analizi daha seyrek çalışabilir


# --- 3. ANA DÖNGÜ (GÖRÜNTÜLEME) ---
def baslat():
    global calisiyor, kare_goruntu  # kare_goruntu eklendi

    # Thread'leri tanımla
    t_yuz = threading.Thread(target=yuz_takibi_islem)
    t_ses = threading.Thread(target=ses_analizi_islem)

    t_yuz.start()
    t_ses.start()

    # OpenCV Penceresi
    while True:
        # Eğer Thread'den görüntü geldiyse onu kullan, gelmediyse siyah ekran göster
        if kare_goruntu is not None:
            ekran = kare_goruntu.copy()
        else:
            ekran = np.zeros((480, 640, 3), dtype=np.uint8)

        # YAZILARI GÖRÜNTÜNÜN ÜZERİNE BİNDİRME İŞLEMİ (HUD - Head Up Display)
        # Alt kısma yarı şeffaf bir bant eklendi
        cv2.rectangle(ekran, (0, 380), (640, 480), (0, 0, 0), -1)

        cv2.putText(
            ekran, f"Goz Durumu: {mesaj_yuz}", (20, 420), 2, 0.7, (0, 255, 0), 1
        )
        cv2.putText(
            ekran, f"Ses Durumu: {mesaj_ses}", (20, 460), 2, 0.7, (255, 255, 0), 1
        )
        cv2.putText(
            ekran, f"TOPLAM PUAN: {int(puan)}", (385, 440), 2, 0.8, (0, 0, 220), 2
        )

        cv2.imshow("AI Interview Coach - v1.0", ekran)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            calisiyor = False
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    baslat()

cv2.destroyAllWindows()
