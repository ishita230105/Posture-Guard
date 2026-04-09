import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

STATE_WAITING = 0
STATE_MONITORING = 1
current_state = STATE_WAITING

good_profile = None
bad_profile = None
baseline_shldr_width = 0
baseline_eye_ratio = 0

def extract_features(landmarks):
    l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    shldr_mid_y = (l_shldr.y + r_shldr.y) / 2
    ear_mid_y = (l_ear.y + r_ear.y) / 2
    
    shldr_width = math.dist([l_shldr.x, l_shldr.y], [r_shldr.x, r_shldr.y])
    
    if shldr_width == 0: return None
    
    neck_length = abs(shldr_mid_y - ear_mid_y) / shldr_width
    head_drop = (nose.y - ear_mid_y) / shldr_width
    
    return np.array([neck_length, head_drop])

print("--- POSTURE MONITOR INITIALIZED ---")
print("1. Sit up straight and press 'g' to save Good Posture.")
print("2. Slouch/Roll your shoulders and press 'b' to save Bad Posture.")
print("3. Press 's' to start monitoring.")

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    current_features = None
    current_shldr_width = 0
    current_eye_ratio = 0

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        current_features = extract_features(landmarks)

        l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        r_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

        current_shldr_width = math.dist([l_shldr.x, l_shldr.y], [r_shldr.x, r_shldr.y])
        if current_shldr_width > 0:
            eye_width = math.dist([l_eye.x, l_eye.y], [r_eye.x, r_eye.y])
            current_eye_ratio = eye_width / current_shldr_width

    if current_state == STATE_WAITING:
        cv2.putText(frame, "SETUP MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if good_profile is not None:
            cv2.putText(frame, "[x] Good Posture Saved", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "[ ] Press 'g' for Good Posture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        if bad_profile is not None:
            cv2.putText(frame, "[x] Bad Posture Saved", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "[ ] Press 'b' for Bad Posture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if good_profile is not None and bad_profile is not None:
            cv2.putText(frame, "Press 's' to START", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    elif current_state == STATE_MONITORING:
        if current_features is not None:
            if current_shldr_width > (baseline_shldr_width * 1.3):
                status = "TOO CLOSE TO SCREEN!"
                color = (0, 165, 255) 
            elif current_eye_ratio < (baseline_eye_ratio * 0.65):
                status = "NECK MOVEMENT"
                color = (255, 255, 0)
            else:
                dist_to_good = np.linalg.norm(current_features - good_profile)
                dist_to_bad = np.linalg.norm(current_features - bad_profile)
                
                if dist_to_bad < dist_to_good:
                    status = "SLOUCH DETECTED!"
                    color = (0, 0, 255)
                else:
                    status = "GREAT POSTURE"
                    color = (0, 255, 0)
                
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    cv2.imshow("PostureGuard AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if current_state == STATE_WAITING:
        if key == ord('g') and current_features is not None:
            good_profile = current_features
            baseline_shldr_width = current_shldr_width
            baseline_eye_ratio = current_eye_ratio
            print("Good posture saved.")
        elif key == ord('b') and current_features is not None:
            bad_profile = current_features
            print("Bad posture saved.")
        elif key == ord('s') and good_profile is not None and bad_profile is not None:
            current_state = STATE_MONITORING
            print("Monitoring started.")
            
    elif current_state == STATE_MONITORING:
        if key == ord('r'):
            current_state = STATE_WAITING
            good_profile = None
            bad_profile = None
            baseline_shldr_width = 0
            baseline_eye_ratio = 0
            print("Resetting calibration.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()