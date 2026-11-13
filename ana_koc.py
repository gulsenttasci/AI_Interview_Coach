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
                
                mesaj_yuz = yuz_takip.analiz_et(yuz.landmark)

                # puanlama
                if "BAKIYORSUN" in mesaj_yuz or "TUT" in mesaj_yuz:
                    puan -= 0.05
        else:
            mesaj_yuz = "YUZ BULUNAMADI!"
            puan -= 0.01  # Hafif puan kırılabilir

        # Görüntüyü ana döngüye gönder
        kare_goruntu = image.copy()

        time.sleep(0.01)

    cap.release()


# ses analiz thread
def ses_analizi_islem():
    global mesaj_ses, puan, calisiyor

    son_soylenen = ""  # Asistanın aynı şeyi üst üste tekrarlamasını önlemek için hafıza

    while calisiyor:

        # cevap geliyorsa arka plan dinlemesi duruyor
        if soru_cevap_modu:
            time.sleep(0.5)
            continue

        # rapor
        rapor, hata_var_mi = ses_koçu.sesi_analiz_et()

        # mesaj güncelleme
        mesaj_ses = rapor

        # puan kontrolü
        if hata_var_mi:
            puan -= 1

        # eğer rapor standart bir mesaj değilse ve aynı cümleyi söylemediyse
        standart_mesajlar = ["SESIN GAYET IYI VE ANLASILIR.", "Dinliyorum, devam et..."]

        if rapor not in standart_mesajlar and rapor != son_soylenen:
            ses_kuyrugu.put(rapor)  # mesajı konuşma kuyruğuna gönder
            son_soylenen = rapor  # söylenen mesajı hafızaya al

        # hafızayı sıfırlama
        if rapor in standart_mesajlar:
            son_soylenen = ""

        time.sleep(0.1)


# apı ile etkileşim
def adayin_cevabini_degerlendir():
    global soru_cevap_modu
    soru_cevap_modu = True  # diğer mikrofono kilitler

    ses_kuyrugu.put("Seni dinliyorum,cevabın nedir?")
    time.sleep(2)  # asistanın susmasını bekler

    r = sr.Recognizer()
    with sr.Microphone() as kaynak:
        try:
            print("CEVAP DİNLENİYOR...")
            r.adjust_for_ambient_noise(kaynak, duration=0.5)
            # phrase_time_kimit=15 :adaya konuşması için 15 saniye sür
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


# tts thread
def konusma_islem():
    global calisiyor

    # ses motorunu başlat
    motor = pyttsx3.init()

    # sesi biraz hızlandırıp profesyonel bir tona alma işlemi (default 200 normalde)
    motor.setProperty("rate", 160)

    # sistemin varsayılan sesi
    sesler = motor.getProperty("voices")
    if len(sesler) > 0:
        motor.setProperty("voice", sesler[0].id)

    while calisiyor:
        # kuyrukta mesaj var mi
        if not ses_kuyrugu.empty():
            okunacak_metin = ses_kuyrugu.get()
            print(f"🤖 Asistan: {okunacak_metin}")
            motor.say(okunacak_metin)
            motor.runAndWait()  #thread bekler konuşma bitene kadar

        time.sleep(0.5)  # işlemciyi yormayalım


#görüntüleme
def baslat():
    global calisiyor, kare_goruntu  # kare_goruntu eklendi

    # Thread'leri tanımlama
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

    # pencere oluşur ve boyutlandırılabilir (gpu!)
    cv2.namedWindow("AI Interview Coach", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Interview Coach", 640, 480)

    print("Kamera hazirlaniyor, lutfen bekleyin...")
    while kare_goruntu is None and calisiyor:
        time.sleep(0.1)

    # OpenCV Penceresi
    while True:
        # eğer Thread'den görüntü geldiyse onu kullan, gelmediyse siyah ekran göster
        if kare_goruntu is not None:
            # görüntüyü ekranı kaplayacak şekilde yeniden boyutlandır
            ekran = cv2.resize(kare_goruntu, (800, 600))
        else:
            ekran = np.zeros((600, 800, 3), dtype=np.uint8)


        cv2.rectangle(ekran, (0, 480), (800, 600), (0, 0, 0), -1)

        #Durum Mesajları
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
            # kamerayı dondurmamak için bu yeni bir Thread olarak çalışır
            threading.Thread(target=adayin_cevabini_degerlendir).start()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    baslat()

cv2.destroyAllWindows()
