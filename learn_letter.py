print(">>> RUNNING UPDATED LEARN_LETTER.PY (NO LANDMARK DRAWING) <<<")

import cv2
import mediapipe as mp
import joblib
import time
from normalization import normalize_hand_landmarks
import os

# ---------------- LOAD MODEL ----------------
model = joblib.load("asl_model.pkl")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- STATE ----------------
target_letter = "A"
last_key_time = 0
KEY_DELAY = 0.3

# ---------------- LOAD IMAGE ----------------
def load_letter_image(letter):
    path = os.path.join("asl_images", f"{letter}.png")
    if os.path.exists(path):
        return cv2.imread(path)
    return None

letter_image = load_letter_image(target_letter)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted = ""

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # 🔴 NO landmark drawing here

        features = normalize_hand_landmarks(hand)
        probs = model.predict_proba([features])[0]
        predicted = model.classes_[probs.argmax()]

    # ---------------- DRAW LETTER IMAGE (BOTTOM LEFT) ----------------
    if letter_image is not None:
        img_h, img_w, _ = letter_image.shape

        x1 = 10
        y1 = h - img_h - 10
        x2 = x1 + img_w
        y2 = y1 + img_h

        if y1 >= 0:
            frame[y1:y2, x1:x2] = letter_image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ---------------- UI HEADER ----------------
    HEADER_HEIGHT = 180
    cv2.rectangle(frame, (0, 0), (w, HEADER_HEIGHT), (0, 0, 0), -1)

    cv2.putText(frame, "ASL LETTER LEARNING MODE",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (220, 220, 220), 2)

    cv2.putText(frame, "Type A-Z to change letter | ESC to quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1)

    cv2.putText(frame, f"Target Letter: {target_letter}",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 255, 0), 3)

    cv2.putText(frame, f"Your Sign: {predicted}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 0), 2)

    cv2.imshow("ASL Letter Learning Mode", frame)

    # ---------------- KEY INPUT ----------------
    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    if 65 <= key <= 90 or 97 <= key <= 122:
        if now - last_key_time > KEY_DELAY:
            target_letter = chr(key).upper()
            letter_image = load_letter_image(target_letter)
            last_key_time = now

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
