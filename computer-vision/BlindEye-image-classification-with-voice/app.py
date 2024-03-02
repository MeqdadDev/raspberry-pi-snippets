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
label = ""
confidence = 0
defined_words = ["exit", "left", "stop", "right"]

def say_label(label):
    tts.say(label)
    tts.runAndWait()

def selectOCR(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    recognized_text = ocr.image_to_string(binary_image).lower()
    if "ex" in recognized_text:
        say_label("exit")
        return True, recognized_text
    elif "st" in recognized_text:
        say_label("stop")
        return True, recognized_text
    elif "le" in recognized_text:
        say_label("left")
        return True, recognized_text
    elif "ri" in recognized_text:
        say_label("right")
        return True, recognized_text
    else:
        return False, recognized_text


while True:
    ret, frame = cap.read()
    if ctr_frm > 23:
        ctr_frm=0
        isText, text = selectOCR(frame)
        print("isText, text", isText, text)
        if isText:
            print(">> OCR:", text, end="\n\n")
            confidence = 0.0
            continue
        cv.imwrite(image_file_name, frame)
        results = tm_model.classify_frame(image_file_name)
        print("Label:", results["label"])
        print("Confidence:", results["confidence"], end="\n\n")
        label = results["label"]
        confidence = results["confidence"]
        if label == "Gray" or label == "Black":
            label += " T shirt"
    else:
        ctr_frm+=1

    if confidence > 97.00:
        cv.putText(frame, f"You see: {label.capitalize()}", (15,40), cv.FONT_ITALIC, 1, (0, 255, 0), 2)

        if (ctr_tts > 30) and "Background" not in label:
            say_label(label)
            ctr_tts=0
        else:
            ctr_tts+=1
    cv.imshow('Blind Eye', frame)
    # cv.putText(frame, f"You {cvText}", (15,40), cv.FONT_ITALIC, 1, (0, 255, 0), 2)

    k = cv.waitKey(1)
    if k%255 == 27:
        break

cap.release()
