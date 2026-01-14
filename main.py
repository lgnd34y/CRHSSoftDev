import cv2
import subprocess
import sys
import numpy as np

# ---------------- BUTTON CLASS ----------------
class Button:
    def __init__(self, x, y, w, h, text, color):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color

    def draw(self, frame):
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h),
                      self.color, -1)
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h),
                      (255, 255, 255), 2)

        text_size = cv2.getTextSize(
            self.text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
        )[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2

        cv2.putText(frame, self.text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2)

    def is_clicked(self, pos):
        return (self.x <= pos[0] <= self.x + self.w and
                self.y <= pos[1] <= self.y + self.h)


# ---------------- MOUSE HANDLING ----------------
clicked_pos = None

def mouse_event(event, x, y, flags, param):
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)


# ---------------- WINDOW SETUP ----------------
WIDTH, HEIGHT = 800, 500
WINDOW_NAME = "ASL Application"

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_event)

# ---------------- BUTTONS ----------------
learn_button = Button(250, 170, 300, 80, "Learn Letters", (0, 170, 0))
word_button  = Button(250, 270, 300, 80, "Word Detection", (0, 120, 200))
quit_button  = Button(250, 370, 300, 60, "Quit", (80, 80, 80))


# ================= MAIN MENU LOOP =================
while True:
    clicked_pos = None

    while True:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)

        # Title
        cv2.putText(frame, "ASL APPLICATION",
                    (215, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (255, 255, 255), 3)

        cv2.putText(frame, "Choose a mode",
                    (305, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (200, 200, 200), 1)

        learn_button.draw(frame)
        word_button.draw(frame)
        quit_button.draw(frame)

        cv2.imshow(WINDOW_NAME, frame)

        if clicked_pos:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            sys.exit()

    # ---------------- HANDLE SELECTION ----------------
    if learn_button.is_clicked(clicked_pos):
        cv2.destroyAllWindows()
        subprocess.run([sys.executable, "learn_letter.py"])
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_event)

    elif word_button.is_clicked(clicked_pos):
        cv2.destroyAllWindows()
        subprocess.run([sys.executable, "predict_words.py"])
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_event)

    elif quit_button.is_clicked(clicked_pos):
        cv2.destroyAllWindows()
        sys.exit()
