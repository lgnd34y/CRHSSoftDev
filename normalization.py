import numpy as np

def normalize_hand_landmarks(hand_landmarks):
      
    pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    pts = np.array(pts, dtype=np.float32)

    for i in range(len(pts)):
        pts[i] = pts[i] - pts[0] 

    hand_span = np.max(np.linalg.norm(pts, axis=1))
    if hand_span > 0:
        pts /= hand_span
    
    theta = np.arctan2(pts[5][1], pts[5][0])  

    cos = np.cos(-theta)
    sin = np.sin(-theta)

    rot_z = np.array([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [0,      0,     1]
    ])

    pts = pts @ rot_z.T  

    return pts.flatten()
