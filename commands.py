import cv2
import mediapipe as mp
import numpy as np
import joblib
import csv
import pyttsx3
from collections import deque, Counter
import time
import config 
import pyvirtualcam
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import normalization
import threading
import queue
import pyttsx3

LAST_TAUGHT_WORD = ""
SHOW_COMMAND_LIST = False
speech_queue = queue.Queue()

def mouse_click_handler(event, x, y, flags, param):
    global SHOW_COMMAND_LIST
    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 <= x <= 630 and 10 <= y <= 50:
            SHOW_COMMAND_LIST = not SHOW_COMMAND_LIST
            
def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 100)
    while True:
        text_to_speak = speech_queue.get()
        if text_to_speak is None: 
            break
        engine.say(text_to_speak)
        engine.runAndWait()

        speech_queue.task_done()

def save_feedback(is_letter, writer, f, current_features, key):
    if is_letter:
        label = chr(key).upper()
    else:
        label = config.word_map[key]
        
    writer.writerow([label] + list(current_features))
    f.flush()
    print(f"Saved Feedback: {label}")

def multiple_letters_screen_text(conf, buffer, model, probs, last_add, text, show_sn, sn_time, sequence_buffer):
    space_was_added = False
    back_was_added = False
    
    if conf > config.confidence_threshold:
        prediction = model.classes_[probs.argmax()]
        buffer.append(prediction)
        
        if len(buffer) == config.max_buffer_length and time.time() - last_add > config.time_between_adds:
            majority = Counter(buffer).most_common(1)[0][0]
            
            if majority == "BACK":
                if len(text) > 0:
                    text = text[:-1]
                    back_was_added = True
            elif majority == "SPACE":
                text += " "
                space_was_added = True
            elif majority == "DELETE":
                text = ""
            else:
                text += majority.replace("_", " ")
            
            last_add = time.time()
            buffer.clear()
            sequence_buffer.clear() 
            
    return text, last_add, show_sn, sn_time, space_was_added, back_was_added

def initialize_frame(cap, hands):
    success = cap.read()[0]
    img = cap.read()[1]
    if not success: 
        return None, None
    frame = cv2.flip(img, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    return frame, results

def start_data_collection():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)

    f = open("asl_data.csv", "a", newline="")
    writer = csv.writer(f)

    return cap, hands, f, writer

def start_prediction():

    with open(config.session_data_path, "w", newline="") as f:
        pass 
    
    model = joblib.load(config.model_path)
    tts = pyttsx3.init()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)
    draw = mp.solutions.drawing_utils
    threading.Thread(target=tts_worker, daemon=True).start()


    buffer = deque(maxlen=config.max_buffer_length)
    sequence_buffer = deque(maxlen=20)
    return model, tts, cap, hands, draw, buffer, sequence_buffer, "", 0, False, 0

def start_active_learning():

    model = joblib.load(config.model_path)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.6)

    f = open(config.feedback_data, "a", newline="")
    writer = csv.writer(f)

    sequence_buffer = deque(maxlen=config.max_sequence_length)
    return model, cap, hands, f, writer, sequence_buffer

def record(results, current_label, writer, sequence_buffer, frame):
    hand = results.multi_hand_landmarks[0]
    hand_type = results.multi_handedness[0].classification[0].label

    feat = normalization.normalize_hand_landmarks(hand, hand_type)
    sequence_buffer.append(feat)

    if len(sequence_buffer) == config.max_sequence_length:
        movement_row = np.array(sequence_buffer).flatten().tolist()
        writer.writerow([current_label] + movement_row)
        print(f"Captured Movement for: {current_label}")
        sequence_buffer.clear() 

    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)        

def predict_letter(results, model, current_features, frame):
    probs = model.predict_proba([current_features])[0]
    conf = max(probs)
    label = model.classes_[probs.argmax()]
    
    hand = results.multi_hand_landmarks[0]
    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)        
    
    cv2.putText(frame, f"Pred: {label} ({conf:.2f})", (10, 50), 
                cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("active_learning",frame)

def find_word(results, model, normalize, buffer, last_add, text, show_sn, sn_time, frame, current_features, sequence_buffer):
    global LAST_TAUGHT_WORD
    probs = None
    conf = 0.0
    space_was_added = False
    back_was_added = False

    if current_features is not None:
        probs = model.predict_proba([current_features])[0]
        if probs is not None and len(probs) > 0:
            conf = np.max(probs) 


    if probs is not None:
        text, last_add, show_sn, sn_time, space_was_added, back_was_added = multiple_letters_screen_text(
            conf, buffer, model, probs, last_add, text, show_sn, sn_time, sequence_buffer
        ) 
    
    return text, last_add, show_sn, sn_time, probs, space_was_added, back_was_added

def predict_word():
    vcam = pyvirtualcam.Camera(width=640, height=480, fps=30)
    model, tts, cap, hands, draw, buffer, seq_buffer, text, last_add, show_sn, sn_time = start_prediction()
    start_y = 95 
    line_height = 35
    win_name = "Private Dashboard"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_click_handler)


    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_features = None 
        if results.multi_hand_landmarks:
            hand_type = results.multi_handedness[0].classification[0].label
            feat = normalization.normalize_hand_landmarks(results.multi_hand_landmarks[0], hand_type)
            
            if feat is not None:

                seq_buffer.append(feat)
                if len(seq_buffer) == 20:
                    current_features = np.array(list(seq_buffer)).flatten()

        text, last_add, show_sn, sn_time, probs, space_was, back_was = find_word(
            results, model, normalization.normalize_hand_landmarks, buffer, 
            last_add, text, show_sn, sn_time, frame, current_features, seq_buffer
        )

        public_frame = frame.copy()
        
        cv2.rectangle(public_frame, (0, 340), (640, 480), (0, 0, 0), -1)
        display_text = text if len(text) < 26 else "..." + text[-23:]
        cv2.putText(public_frame, f"Signed: {display_text}", (15, 395), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255, 255, 255), 2)

        my_preview = public_frame.copy() 

        if len(buffer) > 0:
            current_guess = Counter(buffer).most_common(1)[0][0]
            cv2.putText(my_preview, f"DETECTING: {current_guess}", (20, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(my_preview, (20, 290), (220, 305), (40, 40, 40), -1)
            stab_bar_w = int((len(buffer) / config.max_buffer_length) * 200)
            cv2.rectangle(my_preview, (20, 290), (20 + stab_bar_w, 305), (0, 255, 255), -1)

        cv2.rectangle(my_preview, (500, 10), (630, 50), (100, 100, 100), -1)
        cv2.putText(my_preview, "COMMANDS", (510, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if SHOW_COMMAND_LIST:
            overlay = my_preview.copy()
            cv2.rectangle(overlay, (400, 60), (630, 210), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, my_preview, 0.3, 0, my_preview)
            cv2.putText(my_preview, "R: Teach Sign", (410, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(my_preview, "Enter: Speak", (410, start_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(my_preview, "Bksp: Delete", (410, start_y + (line_height * 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(my_preview, "Esc: Exit", (410, start_y + (line_height * 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        vcam.send(cv2.cvtColor(public_frame, cv2.COLOR_BGR2RGB)) 
        cv2.imshow(win_name, my_preview)

        key = cv2.waitKey(1)
        if key == 27: 
            break
        if key == 8: 
            text = text[:-1]
        if key == 13 and text.strip():
            speech_queue.put(text.strip())
            text = ""
        if key == 114:
            model = teach_new_sign(cap, hands, normalization.normalize_hand_landmarks, model, vcam)
            buffer.clear()
            seq_buffer.clear()

    cap.release()
    vcam.close()
    cv2.destroyAllWindows()

def teach_new_sign(cap, hands, normalization_fn, current_model, vcam):

    global LAST_TAUGHT_WORD
    new_label = input("Type the meaning: ").upper().strip()
    LAST_TAUGHT_WORD = new_label 
    f = open(config.session_data_path, "w", newline="") 
    writer = csv.writer(f)
    
    recorded_samples = []
    temp_buffer = []
    
    privacy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(privacy_frame, "USER IS BUSY", (160, 220), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

    
    while len(recorded_samples) < 30:
        ret = cap.read()[0]
        frame = cap.read()[1]
        if not ret: 
            break
        frame = cv2.flip(frame, 1)
        
        vcam.send(cv2.cvtColor(privacy_frame, cv2.COLOR_BGR2RGB))
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            hand_type = results.multi_handedness[0].classification[0].label
            feat = normalization_fn(results.multi_hand_landmarks[0], hand_type)
            
            if feat is not None:
                temp_buffer.append(feat)
                if len(temp_buffer) == 20:
                    flattened_sequence = np.array(temp_buffer).flatten()
                    recorded_samples.append(flattened_sequence)
                    temp_buffer = [] 
                    

        progress_w = int((len(recorded_samples) / 30) * 300)
        cv2.rectangle(frame, (150, 400), (450, 430), (50, 50, 50), -1)
        cv2.rectangle(frame, (150, 400), (150 + progress_w, 430), (0, 255, 0), -1)
        cv2.putText(frame, f"RECORDING: {new_label}", (180, 380), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Private Dashboard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 113:
            break
        
    if len(recorded_samples) > 0:
            for sample in recorded_samples:
                writer.writerow([new_label] + list(sample))
    

    base_df = pd.read_csv(config.data, header=None)
    session_df = pd.read_csv(config.session_data_path, header=None)
    
    session_df.columns = base_df.columns 
    
    df = pd.concat([base_df, session_df], ignore_index=True).dropna()
    
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].astype(str).values
    
    new_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    new_model.fit(X, y)
    
    return new_model


