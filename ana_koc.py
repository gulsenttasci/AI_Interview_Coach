import cv2
import mediapipe as mp
import threading
import time
import numpy as np
import pyttsx3
import queue
import zeka_api
import speech_recognition as sr
import yuz_takip
import ses_koçu

# --- GLOBAL DEĞİŞKENLER (Thread'ler arası iletişim için) ---
# Shared Memory
soru_cevap_modu = False  # mikrofon çakışmasını önlemek için kilit
mesaj_yuz = "Baslatiliyor..."
mesaj_ses = "Dinleniyor..."
puan = 100
calisiyor = True
kare_goruntu = None  # Kameradan gelen görüntünün saklandığı kısım

ses_kuyrugu = queue.Queue()  # seslendirme için iletişim kuyruğu


# --- 1. MODÜL: YÜZ TAKİBİ THREAD'İ ---
def yuz_takibi_islem():
    global mesaj_yuz, puan, calisiyor, kare_goruntu  # kare_goruntu eklendi
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(1.0)

    if not cap.isOpened():
        print("HATA: Kamera acilamadi! Baska bir uygulama kullaniyor olabilir.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,  # mediapipe default value=0.5
    )

    while calisiyor:
        success, image = cap.read()
        if not success:
            print("Kamera verisi bekleniyor...")
            time.sleep(0.5)  # İşlemciyi kilitlememesi için mola
            continue

        # RAM dostu olması için görüntüyü küçülür fakat gösterirken net kalmalı
        image_small = cv2.resize(image, (320, 240))
        rgb_image = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for yuz in results.multi_face_landmarks:
                # yuz_takip.py
                mesaj_yuz = yuz_takip.analiz_et(yuz.landmark)

                # puanlama
                if "BAKIYORSUN" in mesaj_yuz or "TUT" in mesaj_yuz:
                    puan -= 0.05
        else:
            mesaj_yuz = "YUZ BULUNAMADI!"
            puan -= 0.01  # Hafif puan kırışı

        # Görüntüyü ana döngüye gönder (Kritik Adım)
        kare_goruntu = image.copy()

        # CPU'yu %100 yapmamak için çok kısa bir mola
        time.sleep(0.01)

    cap.release()


# --- 2. MODÜL: SES ANALİZİ THREAD'İ ---
def ses_analizi_islem():
    global mesaj_ses, puan, calisiyor

    son_soylenen = ""  # Asistanın aynı şeyi üst üste tekrarlamasını önlemek için hafıza

    while calisiyor:

        # EĞER ADAY CEVAP VERİYORSA, ARKA PLAN DİNLEMESİNİ DURDUR (Kilit)
        if soru_cevap_modu:
            time.sleep(0.5)
            continue

        # 1. Modülden raporu alma
        rapor, hata_var_mi = ses_koçu.sesi_analiz_et()

        # 2. Mesajı güncelle
        mesaj_ses = rapor

        # 3. Puan kontrolü
        if hata_var_mi:
            puan -= 1

        # 4. Asistan
        # Eğer rapor standart bir mesaj değilse ve aynı cümleyi söylemediyse:
        standart_mesajlar = ["SESIN GAYET IYI VE ANLASILIR.", "Dinliyorum, devam et..."]

        if rapor not in standart_mesajlar and rapor != son_soylenen:
            ses_kuyrugu.put(rapor)  # Mesajı konuşma kuyruğuna gönder
            son_soylenen = rapor  # Söylenen mesajı hafızaya al

        # hafızayı sıfırlama
        if rapor in standart_mesajlar:
            son_soylenen = ""

        time.sleep(0.1)


# ---3.MODÜL: API İLE ETKİLEŞİM THREAD---
def adayin_cevabini_degerlendir():
    global soru_cevap_modu
    soru_cevap_modu = True  # Diğer mikrofono kilitler

    ses_kuyrugu.put("Seni dinliyorum,cevabın nedir?")
    time.sleep(2)  # Asistanın susmasını bekler

    r = sr.Recognizer()
    with sr.Microphone() as kaynak:
        try:
            print("CEVAP DİNLENİYOR...")
            r.adjust_for_ambient_noise(kaynak, duration=0.5)
            # phrase_time_kimit=15 => Adaya konuşması için 15 saniye süre verilir
            audio = r.listen(kaynak, timeout=5, phrase_time_limit=15)

            ses_kuyrugu.put("Cevabını değerlendiriyorum...")

            # sesi metine çevirme
            metin = r.recognize_google(audio, language="tr-TR")
            print(f"CevabıN:{metin}")

            # API'ye gönderme
            yorum = zeka_api.cevabi_degerlendir(metin)
            print(f"Asistan:{yorum}")

            # gelen yorumu asistanın konuşma kuyruğuna ekle
            ses_kuyrugu.put(yorum)

        except sr.WaitTimeoutError:
            ses_kuyrugu.put("Sesini duyamadım,tekrar dener misin?")
        except Exception as e:
            ses_kuyrugu.put("Cevabını tam anlayamadım, tekrar eder misin?")
            print("Hata:", e)

    soru_cevap_modu = False  # kilit açılır, arka plan dinlemesi geri başlar


# --- 4. MODÜL: KONUŞAN KOÇ (TTS) THREAD'İ ---
def konusma_islem():
    global calisiyor

    # Ses motorunu başlat
    motor = pyttsx3.init()

    # Sesi biraz hızlandırıp profesyonel bir tona alma işlemi (Varsayılan 200'dür, 160 ideal)
    motor.setProperty("rate", 160)

    # Sistemin varsayılan sesi
    sesler = motor.getProperty("voices")
    if len(sesler) > 0:
        motor.setProperty("voice", sesler[0].id)

    while calisiyor:
        # Kuyrukta okunacak bir mesaj var mı kontrol et
        if not ses_kuyrugu.empty():
            okunacak_metin = ses_kuyrugu.get()
            print(f"🤖 Asistan: {okunacak_metin}")
            motor.say(okunacak_metin)
            motor.runAndWait()  # Konuşma bitene kadar bu thread'i bekletir (kamerayı etkilemez)

        time.sleep(0.5)  # işlemciyi yormayalım


# --- 3. ANA DÖNGÜ (GÖRÜNTÜLEME) ---
def baslat():
    global calisiyor, kare_goruntu  # kare_goruntu eklendi

    # Thread'leri tanımla
    t_yuz = threading.Thread(target=yuz_takibi_islem)
    t_ses = threading.Thread(target=ses_analizi_islem)
    t_konusma = threading.Thread(target=konusma_islem)  # Yeni Thread

    ses_kuyrugu.put(
        "Merhaba Gulsen. Mulakat provasına hos geldın. Hazırsan baslayalım."
    )

    t_konusma.start()
    t_yuz.start()
    time.sleep(2.0)
    t_ses.start()

    # Pencereyi oluşur ve boyutlandırılabilir (gpu!)
    cv2.namedWindow("AI Interview Coach", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Interview Coach", 640, 480)

    print("Kamera hazirlaniyor, lutfen bekleyin...")
    while kare_goruntu is None and calisiyor:
        time.sleep(0.1)

    # OpenCV Penceresi
    while True:
        # Eğer Thread'den görüntü geldiyse onu kullan, gelmediyse siyah ekran göster
        if kare_goruntu is not None:
            # Görüntüyü ekranı kaplayacak şekilde yeniden boyutlandır
            ekran = cv2.resize(kare_goruntu, (800, 600))
        else:
            ekran = np.zeros((600, 800, 3), dtype=np.uint8)

        # 1.ALT PANEL: YAZILARI GÖRÜNTÜNÜN ÜZERİNE BİNDİRME İŞLEMİ (HUD - Head Up Display)
        # Alt kısma yarı şeffaf bir bant eklendi
        cv2.rectangle(ekran, (0, 480), (800, 600), (0, 0, 0), -1)

        # 2. YAZILARI YENİDEN KONUMLANDIR (Birbirine girmemeleri için aralıkları açıldı)
        # Font büyüklüğünü (0.7) ve kalınlığını (2) optimize edildi

        # SOL TARAF: Durum Mesajları
        cv2.putText(
            ekran,
            f"Goz Durumu: {mesaj_yuz}",
            (30, 525),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            ekran,
            f"Ses Durumu: {mesaj_ses}",
            (30, 570),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            ekran,
            f"TOPLAM PUAN: {int(puan)}",
            (550, 555),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 220),
            3,
        )

        cv2.imshow("AI Interview Coach - v1.1", ekran)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            calisiyor = False
            break
        elif key == ord("c"):  # 'c' tuşuna basıldığında
            # Kamerayı dondurmamak için bu yeni bir Thread olarak çalışır
            threading.Thread(target=adayin_cevabini_degerlendir).start()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    baslat()

cv2.destroyAllWindows()
