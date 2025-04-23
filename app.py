import streamlit as st
import os
import cv2
import tempfile
from src.preprocess import extract_frames, detect_faces_in_frames

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🕵️ Deepfake Detection App")

upload_type = st.radio("Choose input type:", ["Image", "Video"])

if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        file_bytes = uploaded_image.read()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_bytes)
        img_path = tfile.name

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Detect face from image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected.")
        else:
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                st.image(face_img, caption="Detected Face", width=250)
            st.success(f"{len(faces)} face(s) detected.")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        frames_folder = "data/frames_streamlit"
        faces_folder = "data/faces_streamlit"

        extract_frames(video_path, frames_folder, max_frames=20)
        detect_faces_in_frames(frames_folder, faces_folder)

        faces = os.listdir(faces_folder)
        if faces:
            st.success(f"{len(faces)} face(s) detected from video.")
            for face_img in faces[:5]:  # show first 5 faces
                st.image(os.path.join(faces_folder, face_img), width=250)
        else:
            st.warning("No faces detected from video.")
