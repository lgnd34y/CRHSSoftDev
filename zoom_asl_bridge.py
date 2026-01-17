import cv2
import mediapipe as mp
import joblib
import time
import pyttsx3
import pyvirtualcam
from collections import deque, Counter
from normalization import normalize_hand_landmarks  

model = joblib.load("asl_model.pkl")
tts = pyttsx3.init()
tts.setProperty("rate", 160)  

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

buffer = deque(maxlen=12)
text = ""
last_add = 0



vcam = pyvirtualcam.Camera(width=640,height=480,fps=30,print_fps=False)



while True:
    frame = cv2.flip(cap.read()[1], 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    conf = 0.0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        feat = normalize_hand_landmarks(hand)
        probs = model.predict_proba([feat])[0]
        conf = max(probs)
        
        if conf > 0.35: 
            buffer.append(model.classes_[probs.argmax()])
            if len(buffer) == 12 and time.time() - last_add > 1.5:
                text += Counter(buffer).most_common(1)[0][0]
                last_add = time.time()
                buffer.clear()
        else:
            buffer.clear()
    else:
        buffer.clear()  

    
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
    
    cv2.putText(frame, f"Text: {text}", (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(frame, "ENTER = Speak | BACKSPACE = Delete | ESC = Quit", (10, 130), 1, 0.6, (180, 180, 180), 1)


    frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vcam.send(frame_out)

    cv2.imshow("zoom_asl_bridge", frame)

    key = cv2.waitKey(1) 

    if key == 27: 
        break
    if key == 8:  
        text = text[:-1]
    if key == 32: 
        text += " "
    if key == 13 and text.strip():
        tts.say(text)
        tts.runAndWait()


cap.release()
vcam.close()
cv2.destroyAllWindows()
