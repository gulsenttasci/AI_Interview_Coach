import cv2

# Bilgisayarın kamerasını (0 numaralı kamera) başlat
kamera = cv2.VideoCapture(0)

print("Kamera açılıyor... Kapatmak için klavyeden 'q' tuşuna bas!")

while True:
    # Kameradan anlık bir kare oku
    ret, kare = kamera.read()

    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # Görüntüyü ekranda göster
    cv2.imshow("Mülakat Kocu - Test Ekrani", kare)

    # Klavyeden 'q' tuşuna basılırsa döngüyü bitir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kaynağını serbest bırak ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()