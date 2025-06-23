from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import face_recognition
import numpy as np
import pickle
import os
import shutil
from uuid import uuid4

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Directory and Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATABASE_URL = "sqlite:///face_data.db"
Base = declarative_base()

# Database model to store name and face encoding
class FaceEncoding(Base):
    __tablename__ = "encodings"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    encoding = Column(LargeBinary, nullable=False)
    image_filename = Column(String, nullable=True)

# Create DB engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Admin dashboard (HTML interface)
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/image-form", response_class=HTMLResponse)
def image_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/image-login", response_class=HTMLResponse)
def image_login(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register_webcam", response_class=HTMLResponse)
async def show_register(request: Request):
    return templates.TemplateResponse("register_webcam.html", {"request": request})

@app.get("/identify_webcam", response_class=HTMLResponse)
async def show_identify(request: Request):
    return templates.TemplateResponse("identify_webcam.html", {"request": request})

# Identify a person using uploaded image or webcam capture
@app.post("/identify")
async def identify_person(request: Request, file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{file_ext}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        message = "No face detected."
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return {"matched": False, "name": "Unknown", "message": message}
        return templates.TemplateResponse("result.html", {"request": request, "message": message})

    uploaded_encoding = encodings[0]

    try:
        db: Session = SessionLocal()
        faces = db.query(FaceEncoding).all()
        known_encodings = [pickle.loads(f.encoding) for f in faces]
        known_names = [f.name for f in faces]

        matches = face_recognition.compare_faces(known_encodings, uploaded_encoding)
        face_distances = face_recognition.face_distance(known_encodings, uploaded_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]
            message = f"Welcome {name}!"
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                return {"matched": True, "name": name, "message": message}
            return templates.TemplateResponse("result.html", {"request": request, "message": message})
        else:
            message = "Face not recognized."
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                return {"matched": False, "name": "Unknown", "message": message}
            return templates.TemplateResponse("result.html", {"request": request, "message": message})

    except SQLAlchemyError:
        message = "Database error."
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return {"matched": False, "name": "Unknown", "message": message}
        return templates.TemplateResponse("result.html", {"request": request, "message": message})
    finally:
        db.close()

# Register a person using uploaded image or webcam capture
from fastapi.responses import JSONResponse

@app.post("/register")
async def register_person(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid4()}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JSONResponse({"success": False, "message": "No face detected."}, status_code=400)
        return templates.TemplateResponse("result.html", {"request": request, "message": "No face detected."})

    try:
        db: Session = SessionLocal()
        new_face = FaceEncoding(
            name=name,
            encoding=pickle.dumps(encodings[0]),
            image_filename=unique_filename
        )
        db.add(new_face)
        db.commit()

        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JSONResponse({"success": True, "message": f"Registered {name} successfully."})
        return templates.TemplateResponse("result.html", {"request": request, "message": f"Registered {name} successfully."})
    finally:
        db.close()



@app.get("/users", response_class=HTMLResponse)
async def list_users(request: Request):
    try:
        db: Session = SessionLocal()
        users = db.query(FaceEncoding).all()
        return templates.TemplateResponse("users.html", {"request": request, "users": users})
    finally:
        db.close()
@app.post("/delete-user/{user_id}")
async def delete_user(user_id: int):
    try:
        db: Session = SessionLocal()
        user = db.query(FaceEncoding).filter(FaceEncoding.id == user_id).first()
        if user:
            db.delete(user)
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url="/users", status_code=303)

@app.get("/edit-user/{user_id}", response_class=HTMLResponse)
def edit_user_form(request: Request, user_id: int):
    try:
        db: Session = SessionLocal()
        user = db.query(FaceEncoding).filter(FaceEncoding.id == user_id).first()
        if not user:
            return HTMLResponse(content="User not found", status_code=404)
        return templates.TemplateResponse("edit_user.html", {"request": request, "user": user})
    finally:
        db.close()

@app.post("/update-user/{user_id}")
async def update_user_name(user_id: int, name: str = Form(...), file: UploadFile = File(None)):
    try:
        db: Session = SessionLocal()
        user = db.query(FaceEncoding).filter(FaceEncoding.id == user_id).first()

        if not user:
            return HTMLResponse(content="User not found", status_code=404)

        user.name = name

        if file and file.filename:
            file_ext = os.path.splitext(file.filename)[1]
            filename = f"{uuid4()}{file_ext}"
            saved_path = os.path.join(UPLOAD_DIR, filename)

            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            image = face_recognition.load_image_file(saved_path)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                return HTMLResponse(content="No face detected in new image.", status_code=400)

            # Update encoding and filename
            user.encoding = pickle.dumps(encodings[0])
            user.image_filename = filename

        db.commit()
        return RedirectResponse(url="/users", status_code=303)
    finally:
        db.close()

@app.post("/update-face/{user_id}")
async def update_face_via_webcam(user_id: int, file: UploadFile = File(...)):
    db: Session = SessionLocal()
    user = db.query(FaceEncoding).filter(FaceEncoding.id == user_id).first()

    if not user:
        db.close()
        return JSONResponse({"success": False, "message": "User not found"}, status_code=404)

    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid4()}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        db.close()
        return JSONResponse({"success": False, "message": "No face detected"}, status_code=400)

    # Update encoding and image file
    user.encoding = pickle.dumps(encodings[0])
    user.image_filename = filename
    db.commit()
    db.close()

    return JSONResponse({"success": True, "message": "Face updated successfully"})


@app.get("/edit-user-webcam/{user_id}", response_class=HTMLResponse)
async def edit_user_webcam(request: Request, user_id: int):
    db: Session = SessionLocal()
    user = db.query(FaceEncoding).filter(FaceEncoding.id == user_id).first()
    db.close()

    if not user:
        return HTMLResponse(content="User not found", status_code=404)

    return templates.TemplateResponse("edit_user_webcam.html", {"request": request, "user": user})



app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")









'''
# in main.py we have made a simple face recognition system using fastapi here the works only done via uploading images
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import face_recognition
import numpy as np
import pickle
import os
import shutil
from uuid import uuid4

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# here we are defining the path of the encodings.pickle file which is basically used to store the encodings of the faces
# and names of the persons which we have registered in the system
# and also the path of the uploads folder where we will save the uploaded images
# if the uploads folder does not exist then we will create a new folder with the name uploads
# if the encodings.pickle file does not exist then we will create a new file with

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "known_faces", "encodings.pickle")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# here we are defining the path of the enoding.pickle file which is basically used to store the encodings of the faces
# and names of the persons which we have registered in the system
# if the file does not exist then we will create a new file with empty data
# if the file exists then we will load the data from the file
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}


# this will serve an HTML page to let user upload image for register and login from browser
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    # this is the central admin dashboard
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})


# this is an alias route just to render the image-based form again if needed separately
@app.get("/image-form", response_class=HTMLResponse)
def image_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# this is an alias route to render image login form again if needed separately
@app.get("/image-login", response_class=HTMLResponse)
def image_login(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# new route: serves webcam-based registration form (HTML)
@app.get("/register_webcam", response_class=HTMLResponse)
async def show_register(request: Request):
    return templates.TemplateResponse("register_webcam.html", {"request": request})


# new route: serves webcam-based login form (HTML)
@app.get("/identify_webcam", response_class=HTMLResponse)
async def show_identify(request: Request):
    return templates.TemplateResponse("identify_webcam.html", {"request": request})


# this is used to identify the person which we have saved in known_faces/encodings.pickle file
@app.post("/identify")
async def identify_person(request: Request, file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{file_ext}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return templates.TemplateResponse("result.html", {"request": request, "message": "No face detected."})

    uploaded_encoding = encodings[0]
    matches = face_recognition.compare_faces(data["encodings"], uploaded_encoding)
    face_distances = face_recognition.face_distance(data["encodings"], uploaded_encoding)

    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = data["names"][best_match_index]
        return templates.TemplateResponse("result.html", {"request": request, "message": f"Welcome {name}!"})
    else:
        return templates.TemplateResponse("result.html", {"request": request, "message": "Face not recognized."})


# this is used to register the person which we have saved in known_faces/encodings.pickle file
# but we have also made login_via_webcam.py and register_via_webcam.py files
# which are used to register the person via webcam and login the person via same webcam
@app.post("/register")
async def register_person(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{file_ext}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return templates.TemplateResponse("result.html", {"request": request, "message": "No face detected."})

    data["encodings"].append(encodings[0])
    data["names"].append(name)

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    return templates.TemplateResponse("result.html", {"request": request, "message": f"Registered {name} successfully."})


'''
# below code is all working but it is without index.html integration and jinja2 templates

# here we are defining the path of the enoding.pickle file which is basically used to store the encodings of the faces
# and names of the persons which we have registered in the system
# if the file does not exist then we will create a new file with empty data
# if the file exists then we will load the data from the file

"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import pickle
import os
import shutil
from uuid import uuid4

app = FastAPI()

# here we are defining the path of the encodings.pickle file which is basically used to store the encodings of the faces
# and names of the persons which we have registered in the system
# and also the path of the uploads folder where we will save the uploaded images
# if the uploads folder does not exist then we will create a new folder with the name uploads
# if the encodings.pickle file does not exist then we will create a new file with

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "known_faces", "encodings.pickle")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# here we are defining the path of the enoding.pickle file which is basically used to store the encodings of the faces
# and names of the persons which we have registered in the system
# if the file does not exist then we will create a new file with empty data
# if the file exists then we will load the data from the file
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# this is used to identify the person which we have saved in known_faces/encodings.pickle file
@app.post("/identify")
async def identify_person(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{file_ext}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return JSONResponse(content={"message": "No face detected."}, status_code=400)

    uploaded_encoding = encodings[0]

    matches = face_recognition.compare_faces(data["encodings"], uploaded_encoding)
    face_distances = face_recognition.face_distance(data["encodings"], uploaded_encoding)

    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = data["names"][best_match_index]
        return {"matched": True, "name": name}
    else:
        return {"matched": False, "name": "Unknown"}

# this is used to register the person which we have saved in known_faces/encodings.pickle file
# but we have also make login_via_webcam.py and register_via_webcam.py files
# which are used to register the person via webcam and login the person via same webcam
@app.post("/register")
async def register_person(name: str, file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{file_ext}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(saved_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return JSONResponse(content={"message": "No face detected."}, status_code=400)

    new_encoding = encodings[0]

    data["encodings"].append(new_encoding)
    data["names"].append(name)

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    return {"message": f"Registered {name} successfully."}
"""
