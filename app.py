import streamlit as st
import os
from src.preprocess import extract_frames, detect_faces_in_frames
from deepface import DeepFace

st.title("🔍 Deepfake Detection App")

option = st.radio("Select file type:", ("Image", "Video"))

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image("uploaded_image.jpg", caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image uploaded successfully!")

        # Deepfake detection logic
        try:
            analysis = DeepFace.analyze(img_path="uploaded_image.jpg", actions=['emotion', 'age', 'gender'])
            st.write("### 🤖 DeepFace Analysis:")
            st.json(analysis)
        except Exception as e:
            st.error(f"Detection failed: {e}")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video("uploaded_video.mp4")
        st.success("✅ Video uploaded successfully!")
