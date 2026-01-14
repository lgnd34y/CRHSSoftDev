import cv2
import mediapipe as mp
import joblib
import csv
import time
import tkinter as tk
from threading import Thread
from normalization import normalize_hand_landmarks
from collections import deque, Counter

# ---------------- LOAD MODEL ----------------
model = joblib.load("asl_model.pkl")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- ACTIVE LEARNING FILE ----------------
FEEDBACK_FILE = "feedback_data.csv"

# ---------------- SMOOTHING ----------------
PRED_BUFFER = 12
predictions = deque(maxlen=PRED_BUFFER)

current_prediction = ""
current_features = None
last_update_time = 0
ADD_DELAY = 0.8

def most_common(buffer):
    return Counter(buffer).most_common(1)[0][0]

# ---------------- GUI ----------------
def start_gui():
    def yes_clicked():
        if current_prediction and current_features is not None:
            with open(FEEDBACK_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_prediction] + current_features.tolist())
        print("✔ Correct:", current_prediction)

    def no_clicked():
        print("✘ Incorrect. Press correct letter key.")

    root = tk.Tk()
    root.title("ASL Feedback")

    tk.Label(root, text="Is the predicted letter correct?", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="YES", width=20, height=2, command=yes_clicked).pack(pady=5)
    tk.Button(root, text="NO", width=20, height=2, command=no_clicked).pack(pady=5)

    root.mainloop()

# Run GUI in separate thread
Thread(target=start_gui, daemon=True).start()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    letter = ""
    confidence = 0.0
    now = time.time()

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = normalize_hand_landmarks(hand)
        probs = model.predict_proba([features])[0]
        confidence = max(probs)
        letter = model.classes_[probs.argmax()]

        if confidence > 0.3:
            predictions.append(letter)
        else:
            predictions.clear()
    else:
        predictions.clear()

    if len(predictions) == PRED_BUFFER:
        stable_letter = most_common(predictions)

        if now - last_update_time > ADD_DELAY:
            current_prediction = stable_letter
            current_features = features
            last_update_time = now
            predictions.clear()

    # Display
    cv2.putText(frame, f"Prediction: {current_prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Active Learning ASL", frame)

    key = cv2.waitKey(1) & 0xFF

    # If prediction is wrong, user presses correct key
    if 65 <= key <= 90 or 97 <= key <= 122:
        correct_label = chr(key).upper()
        if current_features is not None:
            with open(FEEDBACK_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([correct_label] + current_features.tolist())
            print("✍ Corrected to:", correct_label)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
