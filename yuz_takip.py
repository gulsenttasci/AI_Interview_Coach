import cv2
import mediapipe as mp
import math

#---YARDIMCI HESAPLAMA FONKSİYONU---
def mesafe_hesapla(p1,p2):
    """İki nokta arasındaki 3 boyutlu EUCLİDEAN mesafeyi hesaplar"""
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)

#---ANA ZEKA FONKSİYONU(ana_koc.py beslenme noktası)---
def analiz_et(yuz_noktalari):
    """
    Mediapipe landmark listesini alır ve analiz sonucunu döner.
    """
    # 1. Noktaları Belirleme
    sol_ic = yuz_noktalari[133]
    sol_dis = yuz_noktalari[33]
    sol_bebek = yuz_noktalari[468]

    sag_ic = yuz_noktalari[362]
    sag_dis = yuz_noktalari[263]
    sag_bebek = yuz_noktalari[473]

    # 2. Matematiksel Oranlar
    sol_genislik = mesafe_hesapla(sol_ic, sol_dis)
    sol_bebek_mesafe = mesafe_hesapla(sol_ic, sol_bebek) # Önceki kodda hata vardı, bebeğe olan mesafe olmalı
    sol_oran = sol_bebek_mesafe / (sol_genislik + 0.001)

    sag_genislik = mesafe_hesapla(sag_ic, sag_dis)
    sag_bebek_mesafe = mesafe_hesapla(sag_ic, sag_bebek) # Önceki kodda hata vardı, bebeğe olan mesafe olmalı
    sag_oran = sag_bebek_mesafe / (sag_genislik + 0.001)

    ortalama_oran = (sol_oran + sag_oran) / 2
    z_farki = abs(sol_bebek.z - sag_bebek.z)

    # 3. Karar Mekanizması
    if z_farki > 0.05:
        return "KAFANI DUZ TUTMALISIN"
    elif ortalama_oran < 0.4:
        return "SOLA BAKIYORSUN"
    elif ortalama_oran > 0.6:
        return "SAGA BAKIYORSUN"
    
    return "MUKEMMEL GOZ TEMASI"

if __name__=="__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    kamera = cv2.VideoCapture(0)

    print("Yüz Takip Modülü Test Ediliyor... Çıkış için 'q' basın.")


while True:
    ret, kare = kamera.read()
    if not ret:
        break

    # MediaPipe görüntüleri RGB formatında bekler, kameram ise BGR verir. Dönüştürelim:
    rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

    # Yapay zeka yüzü buluyor...
    sonuclar = face_mesh.process(rgb_kare)

    # Eğer yüz bulunduysa noktaları çiz
    if sonuclar.multi_face_landmarks:
        for yuz in sonuclar.multi_face_landmarks:
            durum=analiz_et(yuz.landmark) #Fonksiyon burada test edilir
            # Ekrana bilgileri yazdırma kısmı
            cv2.putText(kare, durum, (30, 50), cv2.FONT_ITALIC, 1,(0,255,0), 2)
            

    cv2.imshow("Modul Test", kare)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
