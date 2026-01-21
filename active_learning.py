import cv2
import config 
import commands
import normalization
import numpy as np

model, cap, hands, f, writer, sequence_buffer = commands.start_active_learning()

while True:
    frame, results = commands.initialize_frame(cap, hands)
    current_features = None

    if results.multi_hand_landmarks:
        hand_type = results.multi_handedness[0].classification[0].label
        feat = normalization.normalize_hand_landmarks(results.multi_hand_landmarks[0], hand_type)
        sequence_buffer.append(feat)

        if len(sequence_buffer) == config.max_sequence_length:
            current_features = np.array(list(sequence_buffer)).flatten()
            
            commands.predict_letter(results, model, current_features, frame)
                
            key = cv2.waitKey(1) & 0xFF

            if key == 27: 
                f.close()
                break

            if key in config.word_map and current_features is not None:
                commands.save_feedback(is_letter=False, writer=writer, f=f, current_features=current_features, key=key)
            elif 97 <= key <= 122 and current_features is not None:
                commands.save_feedback(is_letter=True, writer=writer, f=f, current_features=current_features, key=key)
    else:
        cv2.imshow("active_learning", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

f.close()
cap.release()
cv2.destroyAllWindows()

