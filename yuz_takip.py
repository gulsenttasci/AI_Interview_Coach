import cv2
import mediapipe as mp

# MediaPipe'ın yüz hattı (Face Mesh) modülünü hazırlayalım
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True) # refine_landmarks gözbebeklerini de takip eder
mp_drawing = mp.solutions.drawing_utils # Noktaları çizmek için yardımcı araç

kamera = cv2.VideoCapture(0)

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
        for yuz_noktalari in sonuclar.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=kare,
                landmark_list=yuz_noktalari,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

    cv2.imshow("Mülakat Koçu - Yapay Zeka Gözü", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()