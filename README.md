Sign2Speak - Real Time ASL to Speech Bridge
The app utilizes MediaPipe for hand tracking and a RandomForestClassifier for gesture recognition. 

Features:
- Hand Skeleton Tracking
- Learning Engine powered by a RandomForest model with 100 decision trees
- Active Learning feature to correct the AI in real time
- Creates a overlay that can be used to connect to Zoom
- Integrated text-to-speech engine that allows the user to speak translated text
- Implements a 20 frame window to capture movement

commands:
git clone https://github.com/yourusername/asl-bridge.git
cd asl-bridge
pip install -r requirements.txt

run python train_model.py to create model, then zoom_asl_bridge.py to run Zoom application

Keyboard Shortcuts
Enter - Triggers Text to Speech
Backspace - Removes the last character from the dashboard
R - Adds a new word to the dictionary
a-z (in active learning mode) saves the current hand pose as that letter

Requirements:
opencv-python
mediapipe
sckit-learn
pandas
pyttsx3
pyvirtualcam
