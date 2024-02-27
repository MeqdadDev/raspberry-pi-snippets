'''
By: MeqdadDev
'''
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv
import pytesseract as ocr
import pyttsx3   #  For Text to Speech

cap = cv.VideoCapture(0)

tts = pyttsx3.init()
tts.setProperty('volume',1.0)

model_path = 'modelFiles/model.tflite'
image_file_name = "frame.jpg"
labels_path = "modelFiles/labels.txt"

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)
ctr_frm = 0
ctr_tts = 0

label = None
confidence = 0

def extract_ocr(img):
    _, binary_image = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow("Binary", binary_image)
    return ocr.image_to_string(binary_image)

def say_label(label, id):
    tts.say(label)
    tts.runAndWait()

while True:
    ret, frame = cap.read()
    if ctr_frm > 20:
        cv.imwrite(image_file_name, frame)
        results = tm_model.classify_frame(image_file_name)
        print("Label:", results["label"])
        print("Confidence:", results["confidence"])
        label = results["label"]
        confidence = results["confidence"]
        id = results["id"]
        ctr_frm=0
    else:
        ctr_frm+=1

    if confidence > 90.00:
        cv.putText(frame, f"You see:{label.capitalize()}", (15,40), cv.FONT_ITALIC, 1, (0, 255, 0), 2)

        if (ctr_tts > 140):
            say_label(label, id)
            ctr_tts=0
        else:
            ctr_tts+=1
    cv.imshow('Blind Eye', frame)
    print(ctr_frm)

    k = cv.waitKey(1)

    if k%255 == 27:
        break
