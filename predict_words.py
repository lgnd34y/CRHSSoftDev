# import cv2
# import mediapipe as mp
# import joblib
# from normalization import normalize_hand_landmarks
# from collections import deque, Counter
# import time
# import pyttsx3

# model = joblib.load("asl_model.pkl")

# tts = pyttsx3.init()
# tts.setProperty("rate", 160)
# tts.setProperty("volume", 1.0)

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     model_complexity=1,
#     min_detection_confidence=0.6,
#     min_tracking_confidence=0.6
# )

# PRED_BUFFER = 12
# predictions = deque(maxlen=PRED_BUFFER)

# current_text = ""
# last_added_time = 0
# ADD_DELAY = 0.8  # delay between letters

# def most_common(buffer):
#     return Counter(buffer).most_common(1)[0][0]

# def speak(text):
#     if text.strip():
#         tts.say(text)
#         tts.runAndWait()

# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     confidence = 0.0
#     now = time.time()

#     if results.multi_hand_landmarks:
#         hand = results.multi_hand_landmarks[0]
#         mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

#         # -------- LETTER PREDICTION --------
#         features = normalize_hand_landmarks(hand)
#         probs = model.predict_proba([features])[0]
#         confidence = max(probs)
#         pred = model.classes_[probs.argmax()]

#         if confidence > 0.35:
#             predictions.append(pred)
#         else:
#             predictions.clear()

#         if len(predictions) == PRED_BUFFER:
#             stable_letter = most_common(predictions)

#             if now - last_added_time > ADD_DELAY:
#                 current_text += stable_letter
#                 last_added_time = now
#                 predictions.clear()

#     else:
#         predictions.clear()

#     # ---------------- UI ----------------
#     cv2.rectangle(frame, (0, 0), (frame.shape[1], 170), (0, 0, 0), -1)

#     cv2.putText(frame, "LETTERS ONLY MODE + TEXT TO SPEECH", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

#     cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

#     cv2.putText(frame, f"Text: {current_text}", (10, 110),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

#     cv2.putText(frame, "ENTER = Speak | BACKSPACE = Delete | ESC = Quit",
#                 (10, 150),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

#     cv2.imshow("ASL Letter Builder (Text to Speech)", frame)

#     key = cv2.waitKey(1) & 0xFF

#     if key == 8:  # BACKSPACE
#         current_text = current_text[:-1]

#     elif key == 13:  # ENTER → SPEAK
#         speak(current_text)

#     elif key == 27:  # ESC
#         break

#     elif key == 32:
#         current_text += " "

# cap.release()
# cv2.destroyAllWindows()
import cv2, mediapipe as mp, joblib, time, pyttsx3
from normalization import normalize_hand_landmarks
from collections import deque, Counter

# 1. SETUP
model = joblib.load("asl_model.pkl")
tts = pyttsx3.init()
tts.setProperty("rate", 160)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

buf = deque(maxlen=12)
text = ""
last_add = 0

while True:
    frame = cv2.flip(cap.read()[1], 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        #draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

        feat = normalize_hand_landmarks(hand)
        probs = model.predict_proba([feat])[0]
        conf, pred = max(probs), model.classes_[probs.argmax()]

        if conf > 0.35:
            buf.append(pred)
            if len(buf) == 12 and (time.time() - last_add > 0.8):
                text += Counter(buf).most_common(1)[0][0]
                last_add = time.time()
                buf.clear()
        else:
            buf.clear()
    else:
        buf.clear()

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
    
    
    # Main Text Display
    cv2.putText(frame, f"Text: {text}", (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(frame, "ENTER = Speak | BACKSPACE = Delete | ESC = Quit", (10, 130), 1, 0.6, (180, 180, 180), 1)

    cv2.imshow("ASL Letter Builder", frame)
    key = cv2.waitKey(1) & 0xFF

    # KEYBOARD CONTROLS
    if key == 27: break
    if key == 8:  text = text[:-1]
    if key == 32: text += " "
    if key == 13 and text.strip():
        tts.say(text)
        tts.runAndWait()

cap.release()
cv2.destroyAllWindows()