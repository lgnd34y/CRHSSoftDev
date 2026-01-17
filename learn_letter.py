import cv2
import mediapipe as mp
import joblib
import string
from normalization import normalize_hand_landmarks

model = joblib.load("asl_model.pkl")
hands = mp.solutions.hands.Hands(max_num_hands=1)

assets = {}
for letter in string.ascii_uppercase:
    assets[letter] = cv2.imread(f"asl_images/{letter}.png")

cap = cv2.VideoCapture(0)
current_letter = "A"

while True:
    frame = cv2.flip(cap.read()[1], 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    height = frame.shape[0]
    width = frame.shape[1]

    prediction = ""

    if results.multi_hand_landmarks:
        landmarks = normalize_hand_landmarks(results.multi_hand_landmarks[0])
        prediction = model.predict([landmarks])[0]
        

   
    ref = assets[current_letter]
    refheight = ref.shape[0] 
    refwidth = ref.shape[1]
    frame[height-refheight-10:height-10, 10:10+refwidth] = ref
    cv2.rectangle(frame, (10, height-refheight-10), (10+refwidth, height-10), (255, 255, 255), 2)

    # 1. Smaller Black Rectangle (Height reduced from 180 to 80)
    cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)

    # 2. Green Text at the very top (y=35)
    cv2.putText(frame, f"Target: {current_letter}", (10, 35), 
                cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 255, 0), 2)

    # 3. Blue Text right under it (y=70)
    # Note: (255, 0, 0) is Blue in OpenCV BGR format
    cv2.putText(frame, f"Your Sign: {prediction}", (10, 70), 
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("learn_letter", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif 97 <= key <= 122:
        current_letter = chr(key).upper()

cap.release()
cv2.destroyAllWindows()