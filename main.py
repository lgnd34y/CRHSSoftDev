import cv2
import subprocess
import sys
import numpy as np


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500

buttons = [
    [250, 150, 300, 70, "Learn Letters", (0, 170, 0), "learn_letter.py"],
    [250, 240, 300, 70, "Word Detection", (0, 120, 200), "predict_words.py"],
    [250, 330, 300, 60, "Quit", (80, 80, 80), None]
]

click_pos = None

def mouse_callback(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def run_script(script_name):
    cv2.destroyAllWindows()
    subprocess.run([sys.executable, script_name])
    cv2.namedWindow("main")
    cv2.setMouseCallback("main", mouse_callback)

cv2.namedWindow("main")
cv2.setMouseCallback("main", mouse_callback)

while True:
    frame = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 20, dtype=np.uint8)

    text_width = cv2.getTextSize("ASL APPLICATION", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0][0]
    text_height = cv2.getTextSize("ASL APPLICATION", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0][1]
    cv2.putText(frame, "ASL APPLICATION", ((SCREEN_WIDTH - text_width) // 2, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    for left_edge, top_edge, width, height, label, color, script in buttons:
        cv2.rectangle(frame, (left_edge, top_edge), (left_edge + width, top_edge + height), color, -1)
        cv2.rectangle(frame, (left_edge, top_edge), (left_edge + width, top_edge + height), (255, 255, 255), 2)
        
        label_width =  cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        label_height =  cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][1]
        cv2.putText(frame, label, (left_edge + (width - label_width) // 2, top_edge + (height + label_height) // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if click_pos:
        click_x = click_pos[0]
        click_y = click_pos[1]
        for left_edge, top_edge, width, height, label, color, script in buttons:
            if left_edge <= click_x <= left_edge + width and top_edge <= click_y <= top_edge + height:
                if label == "Quit": 
                    sys.exit()
                if script: 
                    run_script(script)
        click_pos = None

    cv2.imshow("main", frame)
    
    key = cv2.waitKey(1)
    
    if key == 27: 
        break

cv2.destroyAllWindows()