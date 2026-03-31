import cv2
import mediapipe as mp
import math

# MediaPipe'ın yüz hattı (Face Mesh) modülünü hazırlayalım
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True
)  # refine_landmarks gözbebeklerini de takip eder


kamera = cv2.VideoCapture(0)
mulakat_puani = 100  # Başlangıç puanı


def mesafe_hesapla(p1, p2):
    """İki nokta arasındaki 3 boyutlu Euclidean mesafesini hesaplar."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


while True:
    ret, kare = kamera.read()
    if not ret:
        break

    # MediaPipe görüntüleri RGB formatında bekler, kameramız ise BGR verir. Dönüştürelim:
    rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

    # Yapay zeka yüzü buluyor...
    sonuclar = face_mesh.process(rgb_kare)

    # Eğer yüz bulunduysa noktaları çiz
    if sonuclar.multi_face_landmarks:
        for yuz in sonuclar.multi_face_landmarks:
            # --- SOL GÖZ ANALİZİ---
            sol_ic = yuz.landmark[133]
            sol_dis = yuz.landmark[33]
            sol_bebek = yuz.landmark[468]

            sol_genislik = mesafe_hesapla(sol_ic, sol_dis)
            sol_bebek_mesafe = mesafe_hesapla(sol_ic, sol_dis)
            sol_oran = sol_bebek_mesafe / sol_genislik

            # ---SAĞ GÖZ ANALİZİ---
            sag_ic = yuz.landmark[362]
            sag_dis = yuz.landmark[263]
            sag_bebek = yuz.landmark[473]

            sag_genislik = mesafe_hesapla(sag_ic, sag_dis)
            sag_bebek_mesafe = mesafe_hesapla(sag_ic, sag_dis)
            sag_oran = sag_bebek_mesafe / sag_genislik

            # ---DERİNLİK(Z) KONTROLÜ---
            # İKİ GÖZBEBEĞİ "ARASINDAKİ" DERİNLİK FARKI
            z_farki = abs(sol_bebek.z - sag_bebek.z)

            # ---KARAR MEKANİZMASI---
            ortalama_oran = (sol_oran + sag_oran) / 2

            mesaj = "MUKEMMEL GOZ TEMASI"
            renk = (0, 255, 0)  # yeşil

            if ortalama_oran < 0.4:
                mesaj = "SOLA BAKIYORSUN"
                renk = (0, 0, 255)  # kırmızı
            elif ortalama_oran > 0.6:
                mesaj = "SAGA BAKIYORSUN"
                renk = (0, 0, 255)  # kırmızı again

            if z_farki > 0.05:  # kafanı çok yan çevirdiysen
                mesaj = "KAFANI DUZ TUTMALISIN"
                renk = (0, 165, 255)  # turuncu

            # Ekrana bilgileri yazdırma kısmı
            cv2.putText(kare, mesaj, (30, 50), cv2.FONT_ITALIC, 1, renk, 2)
            cv2.putText(
                kare,
                f"Z Farki: {round(z_farki,3)}",
                (30, 90),
                cv2.FONT_ITALIC,
                0.6,
                (255, 255, 255),
                1,
            )

    cv2.imshow("Mülakat Koçu - 3D Analiz", kare)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
