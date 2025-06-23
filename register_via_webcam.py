# here we register a face via webcam and save the encoding # and name in the encodings.pickle file
# This script captures a face via webcam, encodes it, and saves the encoding along with the user's name to a .pickle file 
# for later authentication. It is used to register new faces.



import cv2
import face_recognition
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, Face  # Assuming `Face` and `Base` are defined in database.py
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

    cv2.imshow("Register Face - Press 'q' to capture", frame)
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

# Get user's name
name = input("Enter your name: ")

# Save to DB
face_encoding = encodings[0]
new_face = Face(name=name, encoding=face_encoding.tobytes())
session.add(new_face)
session.commit()

print(f"[INFO] Face of '{name}' registered successfully to DB.")
