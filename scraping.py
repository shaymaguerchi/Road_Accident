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
sleep_frame_count = 0

# Euclidean distance function
def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def compute_ear(landmarks):
    # Left eye
    top = landmarks[159]
    bottom = landmarks[145]
    left = landmarks[33]
    right = landmarks[133]
    return euclidean(top, bottom) / euclidean(left, right)

# Load video
video_path = 'Lecteur multimédia 2025-04-25 10-50-20.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            ear = compute_ear(face_landmarks.landmark)
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if ear < EAR_THRESHOLD:
                sleep_frame_count += 1
                if sleep_frame_count >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSY", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    engine.say("Attention ! Vous semblez fatigué")
                    engine.runAndWait()
                    sleep_frame_count = 0  # reset after warning
            else:
                sleep_frame_count = 0

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
