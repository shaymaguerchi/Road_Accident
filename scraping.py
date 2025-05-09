import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Voice alert engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# EAR threshold and frame limit
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# Euclidean distance function
def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def compute_ear(landmarks):
    top = landmarks[159]
    bottom = landmarks[145]
    left = landmarks[33]
    right = landmarks[133]
    return euclidean(top, bottom) / euclidean(left, right)

# Load video paths
video_path_face = 'Lecteur multimédia 2025-04-25 10-50-20.mp4'  # Face mesh video
video_path_pose = '#truck #driver #short  #shortsviral #shortsfeed.mp4'  

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Process Face Mesh Video
cap_face = cv2.VideoCapture(video_path_face)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
sleep_frame_count = 0

print("Processing face video...")
while cap_face.isOpened():
    ret, frame = cap_face.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            ear = compute_ear(face_landmarks.landmark)
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if ear < EAR_THRESHOLD:
                sleep_frame_count += 1
                if sleep_frame_count >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSY", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    engine.say("Attention ! Vous semblez fatigué")
                    engine.runAndWait()
                    sleep_frame_count = 0
            else:
                sleep_frame_count = 0

    cv2.imshow('Face Mesh Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_face.release()
cv2.destroyAllWindows()

# Process Pose Video
cap_pose = cv2.VideoCapture(video_path_pose)
pose = mp_pose.Pose()
hand_warning_cooldown = 0
head_warning_cooldown = 0
frame_counter = 0
WARMUP_FRAMES = 10
MIN_HAND_VISIBILITY = 0.5
SHOULDER_VISIBILITY = 0.5

print("Processing pose video...")
while cap_pose.isOpened():
    ret, frame = cap_pose.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter < WARMUP_FRAMES:
        cv2.putText(frame, f"Starting in {WARMUP_FRAMES - frame_counter}...", (30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Hand Position Detection', frame)
        cv2.waitKey(1)
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Get all needed landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

        # --- Head orientation estimation ---
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_offset = nose.x - eye_center_x
        HEAD_TURN_THRESHOLD = 0.05  # Increased threshold for stability

        # Visual debug: draw a line between nose and eye center
        nose_pos = (int(nose.x * w), int(nose.y * h))
        eye_center_pos = (int(eye_center_x * w), int(nose.y * h))
        cv2.line(frame, nose_pos, eye_center_pos, (0, 255, 255), 2)
        cv2.putText(frame, f"Nose offset: {nose_offset:.3f}", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        if abs(nose_offset) > HEAD_TURN_THRESHOLD:
            if head_warning_cooldown <= 0:
                cv2.putText(frame, "FOCUS ON THE ROAD!", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                engine.say("Focus on the road")
                engine.runAndWait()
                head_warning_cooldown = 30  # Reset cooldown
        else:
            head_warning_cooldown = max(0, head_warning_cooldown - 1)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw hand markers
        left_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        cv2.circle(frame, left_coords, 10, (255, 0, 0), -1)
        cv2.circle(frame, right_coords, 10, (0, 0, 255), -1)

        # Hand visibility check
        left_hand_missing = left_wrist.visibility < MIN_HAND_VISIBILITY
        right_hand_missing = right_wrist.visibility < MIN_HAND_VISIBILITY

        if hand_warning_cooldown <= 0 and (left_hand_missing or right_hand_missing):
            cv2.putText(frame, "WARNING: Hand(s) not detected!  FOCUS ON THE ROAD!", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            engine.say("Attention ! Gardez les deux mains sur le volant.")
            engine.runAndWait()
            hand_warning_cooldown = 30
        else:
            hand_warning_cooldown = max(0, hand_warning_cooldown - 1)

        # Debug: visibility scores
        cv2.putText(frame, f"Left: {left_wrist.visibility:.2f}", (30, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Right: {right_wrist.visibility:.2f}", (30, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Hand Position Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_pose.release()
cv2.destroyAllWindows()
