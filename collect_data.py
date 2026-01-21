import cv2
import config
import commands
from collections import deque

cap, hands, f, writer = commands.start_data_collection()
sequnce_buffer = deque(maxlen=config.max_sequence_length)
current_label = None

while True:
    frame, results = commands.initialize_frame(cap, hands)
    
    if results.multi_hand_landmarks and current_label:
        commands.record(results, current_label, writer, sequnce_buffer, frame)

    
    cv2.imshow("collect_data", frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == 27: 
        break
    elif key == 32: 
        current_label = None
    elif key in config.word_map: 
        current_label = config.word_map[key]
    elif 97 <= key <= 122: 
        current_label = chr(key).upper()

f.close()
cap.release()
cv2.destroyAllWindows()