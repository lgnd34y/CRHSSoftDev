import numpy as np

def normalize_hand_landmarks(hand_landmarks):
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    wrist = landmarks[0]
    landmarks -= wrist

    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks /= max_dist

    return landmarks.flatten()


def normalize_two_hands(results):
    """
    Returns a 126-length vector:
    63 for hand 1 + 63 for hand 2
    Missing hands are zero-padded.
    """

    hand_vecs = [
        np.zeros(63, dtype=np.float32),
        np.zeros(63, dtype=np.float32)
    ]

    if not results.multi_hand_landmarks:
        return np.concatenate(hand_vecs)

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        wrist = landmarks[0]
        landmarks -= wrist

        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks /= max_dist

        hand_vecs[i] = landmarks.flatten()

    return np.concatenate(hand_vecs)
