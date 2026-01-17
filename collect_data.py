import cv2
import mediapipe as mp
import csv
from normalization import normalize_hand_landmarks

DATA_FILE = "asl_data.csv"

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

current_label = None
f = open(DATA_FILE, "a", newline="")
writer = csv.writer(f)



while True:
    frame = cv2.flip(cap.read()[1], 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks and current_label:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(

            frame, hand, mp_hands.HAND_CONNECTIONS

        )
        row = normalize_hand_landmarks(hand).tolist()
        writer.writerow([current_label] + row)
        cv2.putText(frame, f"Recording: {current_label}",(10, 40),cv2.FONT_HERSHEY_TRIPLEX,1,(0, 255, 0),2)


    cv2.imshow("collect_data", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        current_label = ""
    elif 97 <= key <= 122:
        current_label = chr(key).upper()

cap.release()
cv2.destroyAllWindows()