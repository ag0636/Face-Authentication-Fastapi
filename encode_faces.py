# here we are encoding the faces of known people
# and saving them in a .pickle file so that we can use them later for authentication
# This script encodes faces from images in the "known_faces" directory and saves them to a .pickle file.
# It is used to register faces for later authentication.
# This script is used to encode faces from images in the "known_faces" directory and save them to a .pickle file.
# It is used to register faces for later authentication.


import os # for file system operations
import face_recognition  # for encoding faces
import pickle  # for saving data to a .pickle file

# Path to known faces
KNOWN_FACES_DIR = "known_faces" # this is the folder which is containing face images which we have load
ENCODINGS_FILE = "encodings.pickle"

known_encodings = []
known_names = []

# Loop over all files in known_faces directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):      
        name = os.path.splitext(filename)[0]  # e.g., "arpit"
        path = os.path.join(KNOWN_FACES_DIR, filename)
#  Loop through all .jpg, .jpeg and .png files
# Extract name from filename (e.g., arpit.jpg → "arpit")

        # Load image and encode face
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
# Load image and get face encoding
# face_encodings() returns a list of face encodings (128-d vector)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"[INFO] Processed {filename}")
# store only the first encoding per pages
        else:
            print(f"[WARNING] No face found in {filename}")

# Save encodings to a file
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("[INFO] Encodings saved to encodings.pickle")
