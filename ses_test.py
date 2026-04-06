import speech_recognition as sr

def sesi_dinle():
    r=sr.Recognizer()
    with sr.Microphone() as kaynak:
        print("Seni dinliyorum...Bir şeyler söyle(Örn: Kendimi tanıtayım)")
        audio=r.listen(kaynak)


        try:
            metin=r.recognize_google(audio,language="tr-TR")
            print(f"Söylediğin:{metin}")

            #Basit bir ton/duygu analizi mantığı
            if"heyecan" in metin.lower():
                print("Analiz: Biraz stresli görünüyorsun,derin bir nefes al ve gevşe :)")
            else:
                print("Analiz: Ses tonun dengeli gözüküyor.")


        except Exception as e:
            print("Sesin anlaşılmıyor,mikrofonunu kontrol et.")

sesi_dinle()                
