import cv2

kamera = cv2.VideoCapture(0)

print("Kamera açılıyor... Kapatmak için klavyeden 'q' tuşuna bas!")

while True:
    #anlık bir kare oku
    ret, kare = kamera.read()

    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    #ekranda göster
    cv2.imshow("Mülakat Kocu - Test Ekrani", kare)

    #'q' tuşuna basılırsa döngüyü bitir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kaynağını serbest bırak ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()