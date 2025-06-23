# here we will implement a simple face authentication system using webcam input
# This script captures video from the webcam, detects faces, and compares them with known encodings
# to authenticate users. It uses the face_recognition library for face detection and encoding.
# It requires the OpenCV library for video capture and display.
# Ensure you have the required libraries installed:
# pip install opencv-python face_recognition
# webcam_auth.py
# This script captures video from the webcam, detects faces, and compares them with known encodings
# to authenticate users. It uses the face_recognition library for face detection and encoding.

import cv2
import face_recognition
import pickle

# Load known face encodings and names
with open("known_faces/encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Start the webcam
print("[INFO] Starting video stream. Press 'q' to quit.")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    # Resize frame for faster processing (optional)
    # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = []

    # Compute encodings for each detected face
    for face_location in face_locations:
        encoding = face_recognition.face_encodings(rgb_frame, [face_location])
        if encoding:
            face_encodings.append(encoding[0])

    # Loop through each detected face and compare with known encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = data["names"][matched_idx]

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow("Face Authentication", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
