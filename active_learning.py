import cv2, mediapipe as mp, joblib, csv
from normalization import normalize_hand_landmarks

model = joblib.load("asl_model.pkl")
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)
current_letter = ""
current_features = None
f = open("feedback_data.csv", "a", newline="")
writer = csv.writer(f)

while True:
    frame = cv2.flip(cap.read()[1], 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        current_features = normalize_hand_landmarks(results.multi_hand_landmarks[0])
        probs = model.predict_proba([current_features])[0]
        conf = max(probs)

        mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{model.classes_[probs.argmax()]} ({conf:.2f})", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("active_learning", frame)
    key = cv2.waitKey(1) 

    if key != -1 and current_features is not None:
        if key == 27: 
            f.close()
            break
        
        label = chr(key).upper()

        if 97 <= key <= 122:
            writer.writerow([label] + list(current_features))
            print(f"Saved: {label}")

cap.release()
cv2.destroyAllWindows()