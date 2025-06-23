# here we implement a simple face recognition login system using webcam input.
# This script captures video from the webcam, detects faces, and compares them with known encodings
# to authenticate users. It uses the face_recognition library for face detection and encoding.
# It requires the OpenCV library for video capture and display.
# Ensure you have the required libraries installed:
# pip install opencv-python face_recognition
# login_via_webcam.py


import cv2
import face_recognition
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Face  # Assuming Face model is defined in database.py
import os

# DB setup
engine = create_engine("sqlite:///face_data.db")
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# Start webcam
print("[INFO] Starting webcam. Press 'q' to capture.")
video_stream = cv2.VideoCapture(0)

while True:
    ret, frame = video_stream.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    cv2.imshow("Login Face - Press 'q' to capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()

# Convert to RGB and encode face
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(rgb_frame)
encodings = face_recognition.face_encodings(rgb_frame, face_locations)

if not encodings:
    print("[ERROR] No face detected. Try again.")
    exit()

input_encoding = encodings[0]

# Load all known encodings from DB
faces = session.query(Face).all()
if not faces:
    print("[ERROR] No faces found in the database.")
    exit()

known_encodings = [np.frombuffer(face.encoding, dtype=np.float64) for face in faces]
known_names = [face.name for face in faces]

# Compare
matches = face_recognition.compare_faces(known_encodings, input_encoding)
face_distances = face_recognition.face_distance(known_encodings, input_encoding)

if True in matches:
    best_match_index = np.argmin(face_distances)
    name = known_names[best_match_index]
    print(f"[INFO] Access granted. Welcome back, {name}!")
else:
    print("[WARNING] Access denied. Face not recognized.")
