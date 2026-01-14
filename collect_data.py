import cv2
import mediapipe as mp
import csv
from normalization import normalize_hand_landmarks

DATA_FILE = "asl_data.csv"

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # 🔹 ONE HAND ONLY
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

current_label = None

with open(DATA_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            if current_label:
                features = normalize_hand_landmarks(hand)
                row = [current_label] + features.tolist()
                writer.writerow(row)

                cv2.putText(
                    frame,
                    f"Recording: {current_label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            frame,
            "Press A-Z to label | ESC to quit",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("ASL One-Hand Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif 65 <= key <= 90:
            current_label = chr(key)
        elif 97 <= key <= 122:
            current_label = chr(key).upper()

cap.release()
cv2.destroyAllWindows()
