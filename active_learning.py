import cv2
import mediapipe as mp
import joblib
import csv
from normalization import normalize_hand_landmarks
from collections import deque, Counter
import time

model = joblib.load("asl_model.pkl")
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)

current_letter = ""
current_features = None

f = open("feedback_data.csv", "a", newline="")
writer = csv.writer(f)

last_add = 0
buffer = deque(maxlen=12)

while True:
    frame = cv2.flip(cap.read()[1], 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        feat = normalize_hand_landmarks(hand)
        probs = model.predict_proba([feat])[0]
        conf = max(probs)

        mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{model.classes_[probs.argmax()]} ({conf:.2f})", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 2)

        if conf > 0.35:
            buffer.append(model.classes_[probs.argmax()])
            if len(buffer) == 12 and time.time() - last_add > 0.8:
                current_letter= Counter(buffer).most_common(1)[0][0]
                current_features = feat
                last_add = time.time()
                buffer.clear()
        else:
            buffer.clear()


    cv2.imshow("active_learning", frame)
    
    key = cv2.waitKey(1) 

    if key != -1 and current_features is not None:
        if key == 27: 
            f.close()
            break
        
        if 97 <= key <= 122:
            label = chr(key).upper()
            writer.writerow([label] + list(current_features))
            print(f"Saved: {label}")

cap.release()
cv2.destroyAllWindows()

