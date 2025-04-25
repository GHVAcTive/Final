# app.py

import streamlit as st
import os
from PIL import Image
from collections import Counter
from deepface import DeepFace
from src.preprocess import extract_frames, detect_faces_in_frames
from src.model import load_deepfake_model, predict_deepfake

# ====================== SETUP ======================
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🔍 Deepfake Detection App")

option = st.radio("Select file type:", ("Image", "Video"))

# Create folders
os.makedirs("uploaded_files", exist_ok=True)
os.makedirs("uploaded_files/frames", exist_ok=True)
os.makedirs("uploaded_files/faces", exist_ok=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model_once():
    return load_deepfake_model("models/deepfake_detector.h5")

model = load_model_once()

# ====================== IMAGE HANDLING ======================
if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_path = os.path.join("uploaded_files", "uploaded_image.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        st.image(image_path, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image uploaded successfully!")

        try:
            st.write("🧠 Running Deepfake Classifier...")
            label, confidence = predict_deepfake(model, image_path)
            st.success(f"🎯 **Prediction:** {label} ({confidence*100:.2f}% confidence)")
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")

# ====================== VIDEO HANDLING ======================
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        video_path = os.path.join("uploaded_files", "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)
        st.success("✅ Video uploaded successfully!")

        # Step 1: Extract frames
        st.write("📤 Extracting frames from video...")
        extract_frames(video_path, "uploaded_files/frames", max_frames=10)

        # Step 2: Detect faces in frames
        st.write("🧠 Detecting faces in frames...")
        detect_faces_in_frames("uploaded_files/frames", "uploaded_files/faces")

        # Step 3: Classify each face
        face_folder = "uploaded_files/faces"
        predictions = []

        st.write("🧪 Classifying detected faces...")
        for filename in os.listdir(face_folder):
            if filename.endswith(".jpg"):
                face_path = os.path.join(face_folder, filename)
                label, confidence = predict_deepfake(model, face_path)
                predictions.append(label)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(Image.open(face_path), caption=filename, width=100)
                with col2:
                    st.write(f"**Prediction:** {label} ({confidence*100:.2f}%)")

        # Step 4: Final video-level result
        if predictions:
            final = Counter(predictions).most_common(1)[0][0]
            st.success(f"🎯 **Final Video Classification:** {final}")
        else:
            st.warning("⚠️ No faces detected in the video.")
