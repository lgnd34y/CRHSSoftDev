import numpy as np

def normalize_hand_landmarks(hand_landmarks, handedness):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    landmarks -= landmarks[0]

    angle = np.arctan2(landmarks[9, 0], landmarks[9, 1])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    landmarks[:, :2] = np.dot(landmarks[:, :2], rotation_matrix)

    dist = np.max(np.linalg.norm(landmarks, axis=1))
    if dist > 0:
        landmarks /= dist

    return landmarks.flatten()

def get_movement_features(current_landmarks, previous_landmarks):
    velocity = current_landmarks - previous_landmarks
    return np.hstack([current_landmarks, velocity])